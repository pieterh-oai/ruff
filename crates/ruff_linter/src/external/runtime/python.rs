#[cfg(not(Py_GIL_DISABLED))]
compile_error!(
    "The external runtime now assumes a free-threaded CPython build. \
     Rebuild PyO3 with `UNSAFE_PYO3_BUILD_FREE_THREADED=1` so that `Py_GIL_DISABLED` is set."
);
thread_local! {
    static PY_SESSION_DEPTH: Cell<usize> = const { Cell::new(0) };
}

use std::cell::{Cell, RefCell};
use std::ffi::CString;
use std::fmt;
use std::hash::Hasher;
use std::sync::{Arc, Mutex, OnceLock};

use crate::checkers::ast::Checker;
use crate::external::RuleLocator;
use crate::external::ast::python::store::{AstStoreHandle, with_store};
use crate::external::ast::python::{
    ModuleTypes, ProjectionTypesRef, expr_to_python, load_module_types, source, stmt_to_python,
};
use crate::external::ast::registry::ExternalLintRegistry;
use crate::external::ast::target::{AstTarget, ExprKind, StmtKind};
use crate::external::error::ExternalLinterError;
use crate::warn_user;
use pyo3::conversion::IntoPyObject;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use ruff_cache::{CacheKey, CacheKeyHasher};
use ruff_python_ast::name::{QualifiedName, UnqualifiedName};
use ruff_python_ast::{Expr, Stmt};
use ruff_python_semantic::NodeId;
use ruff_source_file::SourceFile;
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap, FxHashSet};
use toml::value::{Table as TomlTable, Value as TomlValue};

pub(crate) use semantic::SemanticModelView;

thread_local! {
    static RUNTIME_CACHE: RefCell<FxHashMap<u64, RegistryRuntime>> =
        RefCell::new(FxHashMap::default());
}

static VERIFIED_REGISTRIES: OnceLock<Mutex<FxHashSet<u64>>> = OnceLock::new();

#[derive(Debug)]
pub(crate) struct ExternalRuleHandle {
    check_stmt: Option<Py<PyAny>>,
    check_expr: Option<Py<PyAny>>,
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeEnvironment {
    module_types: Arc<ModuleTypes>,
    linter_configs: Arc<FxHashMap<String, PyObject>>,
}

impl RuntimeEnvironment {
    fn new(module_types: ModuleTypes, linter_configs: FxHashMap<String, PyObject>) -> Self {
        Self {
            module_types: Arc::new(module_types),
            linter_configs: Arc::new(linter_configs),
        }
    }

    fn module_types(&self) -> Arc<ModuleTypes> {
        Arc::clone(&self.module_types)
    }

    fn linter_config(&self, py: Python<'_>, id: &str) -> PyObject {
        let Some(config) = self.linter_configs.get(id) else {
            return PyDict::new(py).into();
        };
        // We could return `dict.copy()` here to guard against user mutation leaking across callbacks;
        // for now we rely on callers not to mutate the cached config.
        config.clone_ref(py)
    }
}

pub(crate) type CompiledCodeMap = FxHashMap<RuleLocator, ExternalRuleHandle>;

#[derive(Clone)]
pub(crate) struct ExternalLintRuntime {
    registry: Arc<ExternalLintRegistry>,
    runtime_cache: RuntimeCache,
}

impl ExternalLintRuntime {
    pub(crate) fn new(registry: ExternalLintRegistry) -> Self {
        let mut hasher = CacheKeyHasher::new();
        registry.cache_key(&mut hasher);
        let pool_id = hasher.finish();
        ensure_registry_verified(pool_id, &registry);

        let registry = Arc::new(registry);
        Self {
            runtime_cache: RuntimeCache::new(pool_id),
            registry,
        }
    }

    pub(crate) fn registry(&self) -> &ExternalLintRegistry {
        self.registry.as_ref()
    }

    pub(crate) fn run_on_stmt(&self, checker: &Checker<'_>, stmt: &Stmt) {
        self.run_on_stmt_with_kind(checker, StmtKind::from(stmt), stmt);
    }

    pub(crate) fn run_on_expr(&self, checker: &Checker<'_>, expr: &Expr) {
        self.run_on_expr_with_kind(checker, ExprKind::from(expr), expr);
    }

    pub(crate) fn run_on_function_def_deferred(&self, checker: &Checker<'_>, stmt: &Stmt) {
        self.run_on_stmt_with_kind(checker, StmtKind::FunctionDefDeferred, stmt);
    }

    pub(crate) fn run_on_lambda_deferred(&self, checker: &Checker<'_>, expr: &Expr) {
        self.run_on_expr_with_kind(checker, ExprKind::LambdaDeferred, expr);
    }

    #[allow(clippy::unused_self)]
    pub(crate) fn run_in_session<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        with_attached_python(|_| f())
    }

    fn run_on_stmt_with_kind(&self, checker: &Checker<'_>, kind: StmtKind, stmt: &Stmt) {
        let locators: Vec<_> = self.registry.rules_for_stmt(kind).collect();
        self.dispatch_rules(
            checker,
            stmt,
            &locators,
            |_| true,
            |handle| handle.check_stmt.as_ref(),
            stmt_to_python,
        );
    }

    fn run_on_expr_with_kind(&self, checker: &Checker<'_>, kind: ExprKind, expr: &Expr) {
        let locators: Vec<_> = self.registry.rules_for_expr(kind).collect();
        let mut call_callee_cache = CachedCallee::default();
        self.dispatch_rules(
            checker,
            expr,
            &locators,
            |rule| rule_applicable_to_expr(rule, kind, expr, &mut call_callee_cache),
            |handle| handle.check_expr.as_ref(),
            expr_to_python,
        );
    }

    fn dispatch_rules<'node, Node>(
        &self,
        checker: &Checker<'_>,
        node: &'node Node,
        locators: &[RuleLocator],
        mut should_run: impl FnMut(&crate::external::ast::rule::ExternalAstRule) -> bool,
        get_callback: impl Fn(&ExternalRuleHandle) -> Option<&Py<PyAny>>,
        convert: impl Fn(
            Python<'_>,
            &crate::Locator<'_>,
            &'node Node,
            ProjectionTypesRef,
        ) -> PyResult<PyObject>,
    ) where
        Node: Ranged + 'node,
    {
        if locators.is_empty() {
            return;
        }

        with_attached_python(|py| {
            self.runtime_cache
                .with_runtime(self.registry.as_ref(), |environment, compiled| {
                    let module_types = environment.module_types();
                    let context = ExternalCheckerContext::new(checker, module_types.projection);
                    context.with_checker_context(|| {
                        for &rule_locator in locators {
                            let (linter, rule) = self.registry.expect_entry(rule_locator);
                            if !should_run(rule) {
                                continue;
                            }
                            if let Some(handle) = compiled.get(&rule_locator) {
                                if let Some(callback) = get_callback(handle) {
                                    let result = (|| -> PyResult<()> {
                                        let py_node = convert(
                                            py,
                                            checker.locator(),
                                            node,
                                            context.projection(),
                                        )?;
                                        let config = environment.linter_config(py, &linter.id);
                                        let runtime_context = build_context(
                                            py,
                                            module_types.as_ref(),
                                            &config,
                                            rule,
                                            node.range(),
                                            checker,
                                        )?;
                                        let _semantic_handle =
                                            SemanticHandle::new(&runtime_context, py);
                                        let outcome = callback
                                            .call1(py, (py_node, runtime_context.context(py)));
                                        runtime_context.flush(py, checker);
                                        outcome.map(|_| ())
                                    })();
                                    if let Err(err) = result {
                                        self.report_python_error(py, rule_locator, &err);
                                    }
                                }
                            }
                        }
                    });
                });
        });
    }

    fn report_python_error(&self, py: Python<'_>, locator: RuleLocator, err: &PyErr) {
        let (linter, rule) = self.registry.expect_entry(locator);
        warn_user!(
            "Error while executing external rule `{}` in linter `{}`: {err}",
            rule.code.as_str(),
            linter.id
        );
        err.print(py);
    }
}

impl fmt::Debug for ExternalLintRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ExternalLintRuntime");
        debug.field("registry", &self.registry);
        debug.field("runtime_cache", &self.runtime_cache);
        debug.finish()
    }
}

struct RegistryRuntime {
    environment: RuntimeEnvironment,
    compiled: CompiledCodeMap,
}

impl fmt::Debug for RegistryRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegistryRuntime")
            .field("environment", &"RuntimeEnvironment")
            .field("compiled_rules", &self.compiled.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug)]
struct RuntimeCache {
    id: u64,
}

impl RuntimeCache {
    fn new(id: u64) -> Self {
        Self { id }
    }

    fn with_runtime<F, R>(self, registry: &ExternalLintRegistry, f: F) -> R
    where
        F: FnOnce(&RuntimeEnvironment, &CompiledCodeMap) -> R,
    {
        RUNTIME_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let entry = cache.entry(self.id).or_insert_with(|| {
                create_registry_runtime(registry).unwrap_or_else(|error| {
                    panic!("failed to initialize external linter runtime: {error}")
                })
            });
            f(&entry.environment, &entry.compiled)
        })
    }
}

fn create_registry_runtime(
    registry: &ExternalLintRegistry,
) -> Result<RegistryRuntime, ExternalLinterError> {
    let (module_types, compiled, linter_configs) = compile_scripts(registry)?;
    let environment = RuntimeEnvironment::new(module_types, linter_configs);
    Ok(RegistryRuntime {
        environment,
        compiled,
    })
}

fn ensure_python_initialized() {
    static PYTHON_INIT: OnceLock<()> = OnceLock::new();
    PYTHON_INIT.get_or_init(|| {
        pyo3::prepare_freethreaded_python();
    });
}

#[allow(unsafe_code)]
fn with_attached_python<F, R>(f: F) -> R
where
    F: for<'py> FnOnce(Python<'py>) -> R,
{
    struct DepthGuard<'a>(&'a Cell<usize>);
    impl Drop for DepthGuard<'_> {
        fn drop(&mut self) {
            let current = self.0.get();
            debug_assert!(current > 0);
            self.0.set(current - 1);
        }
    }

    ensure_python_initialized();

    PY_SESSION_DEPTH.with(|depth| {
        if depth.get() > 0 {
            unsafe { f(Python::assume_gil_acquired()) }
        } else {
            Python::with_gil(|py| {
                depth.set(1);
                let _guard = DepthGuard(depth);
                f(py)
            })
        }
    })
}

fn compile_scripts(
    registry: &ExternalLintRegistry,
) -> Result<(ModuleTypes, CompiledCodeMap, FxHashMap<String, PyObject>), ExternalLinterError> {
    with_attached_python(|py| -> Result<_, ExternalLinterError> {
        let module_types =
            load_module_types(py).map_err(|err| ExternalLinterError::ScriptCompile {
                message: format!("failed to initialize external runtime module: {err}"),
            })?;

        let mut compiled = FxHashMap::default();
        let mut errors = Vec::new();

        for locator in registry.iter_enabled_rule_locators() {
            let (linter, rule) = registry.expect_entry(locator);
            // `PyModule::from_code` uses the module name to populate `sys.modules`. Ensure that
            // the name is unique per script to avoid cross-rule (and cross-test) contamination
            // when scripts are compiled in parallel.
            let script_hash = {
                let mut hasher = CacheKeyHasher::new();
                let path_str = rule.script.path().to_string_lossy();
                hasher.write_usize(path_str.len());
                hasher.write(path_str.as_bytes());
                let contents_str = rule.script.body();
                hasher.write_usize(contents_str.len());
                hasher.write(contents_str.as_bytes());
                hasher.finish()
            };
            let module_name = format!(
                "ruff_external_{}_{}_{script_hash:016x}",
                linter.id,
                rule.code.as_str()
            );
            let Ok(code_cstr) = CString::new(rule.script.body()) else {
                errors.push(ExternalLinterError::format_script_compile_message(
                    linter.id.as_ref(),
                    rule.code.as_str(),
                    Some(rule.script.path().to_path_buf()),
                    "script body contains an interior NUL byte",
                ));
                continue;
            };
            let file_name_owned = rule.script.path().to_string_lossy();
            let Ok(file_cstr) = CString::new(file_name_owned.as_ref()) else {
                errors.push(ExternalLinterError::format_script_compile_message(
                    linter.id.as_ref(),
                    rule.code.as_str(),
                    Some(rule.script.path().to_path_buf()),
                    "script path contains an interior NUL byte",
                ));
                continue;
            };
            let Ok(module_cstr) = CString::new(module_name) else {
                errors.push(ExternalLinterError::format_script_compile_message(
                    linter.id.as_ref(),
                    rule.code.as_str(),
                    Some(rule.script.path().to_path_buf()),
                    "module name contains an interior NUL byte",
                ));
                continue;
            };
            match PyModule::from_code(
                py,
                code_cstr.as_c_str(),
                file_cstr.as_c_str(),
                module_cstr.as_c_str(),
            ) {
                Ok(module) => match build_rule_handle(&module, linter.id.as_ref(), rule) {
                    Ok(handle) => {
                        compiled.insert(locator, handle);
                    }
                    Err(err) => errors.push(err),
                },
                Err(err) => errors.push(ExternalLinterError::format_script_compile_message(
                    linter.id.as_ref(),
                    rule.name.as_ref(),
                    Some(rule.script.path().to_path_buf()),
                    err.to_string(),
                )),
            }
        }

        if !errors.is_empty() {
            return Err(ExternalLinterError::ScriptCompile {
                message: errors.join("\n"),
            });
        }

        let linter_configs = build_linter_configs(py, registry).map_err(|err| {
            ExternalLinterError::ScriptCompile {
                message: format!("failed to convert external linter configuration: {err}"),
            }
        })?;

        Ok((module_types, compiled, linter_configs))
    })
}

fn build_linter_configs(
    py: Python<'_>,
    registry: &ExternalLintRegistry,
) -> PyResult<FxHashMap<String, PyObject>> {
    let mut configs = FxHashMap::default();
    for linter in registry.linters() {
        let config = config_to_python(py, linter.config())?;
        configs.insert(linter.id.clone(), config);
    }
    Ok(configs)
}

fn build_rule_handle(
    module: &Bound<'_, PyModule>,
    linter: &str,
    rule: &crate::external::ast::rule::ExternalAstRule,
) -> Result<ExternalRuleHandle, String> {
    let needs_stmt = rule
        .targets
        .iter()
        .any(|target| matches!(target, AstTarget::Stmt(_)));
    let needs_expr = rule
        .targets
        .iter()
        .any(|target| matches!(target, AstTarget::Expr(_)));

    let check_stmt = lookup_callable(module, "check_stmt");
    let check_expr = lookup_callable(module, "check_expr");

    if needs_stmt && check_stmt.is_none() {
        return Err(ExternalLinterError::MissingHandler {
            linter: linter.to_string(),
            rule: rule.name.clone(),
            handler: "check_stmt".to_string(),
            target: "stmt".to_string(),
        }
        .to_string());
    }

    if needs_expr && check_expr.is_none() {
        return Err(ExternalLinterError::MissingHandler {
            linter: linter.to_string(),
            rule: rule.name.clone(),
            handler: "check_expr".to_string(),
            target: "expr".to_string(),
        }
        .to_string());
    }

    Ok(ExternalRuleHandle {
        check_stmt,
        check_expr,
    })
}

fn lookup_callable(module: &Bound<'_, PyModule>, name: &str) -> Option<Py<PyAny>> {
    match module.getattr(name) {
        Ok(value) if value.is_callable() => Some(value.into_any().unbind()),
        _ => None,
    }
}

struct RuntimeContext {
    context: PyObject,
    reporter: Py<PyReporter>,
    semantic: Py<SemanticModelView>,
}

/// Ensures the semantic model is deactivated even if a callback unwinds early.
struct SemanticHandle<'py> {
    py: Python<'py>,
    context: &'py RuntimeContext,
}

impl<'py> SemanticHandle<'py> {
    fn new(context: &'py RuntimeContext, py: Python<'py>) -> Self {
        Self { py, context }
    }
}

impl Drop for SemanticHandle<'_> {
    fn drop(&mut self) {
        self.context.deactivate_semantic(self.py);
    }
}

impl RuntimeContext {
    fn context(&self, py: Python<'_>) -> PyObject {
        self.context.clone_ref(py)
    }

    fn flush(&self, py: Python<'_>, checker: &Checker<'_>) {
        let reporter = self.reporter.bind(py);
        reporter.borrow().drain_into(checker);
    }

    fn deactivate_semantic(&self, py: Python<'_>) {
        let bound = self.semantic.bind(py);
        bound.borrow().deactivate();
    }
}

/// Per-checker state used when converting nodes and building contexts.
struct ExternalCheckerContext {
    store: AstStoreHandle,
    source_file: SourceFile,
    projection: ProjectionTypesRef,
}

impl ExternalCheckerContext {
    fn new(checker: &Checker<'_>, projection: ProjectionTypesRef) -> Self {
        let source_file = checker.owned_source_file();
        Self {
            store: AstStoreHandle::new(),
            source_file,
            projection,
        }
    }

    fn projection(&self) -> ProjectionTypesRef {
        self.projection
    }

    fn store(&self) -> AstStoreHandle {
        self.store.clone()
    }

    fn source_file(&self) -> &SourceFile {
        &self.source_file
    }

    fn with_checker_context<R>(&self, f: impl FnOnce() -> R) -> R {
        with_store(self.store(), || {
            source::with_source_file(self.source_file(), f)
        })
    }
}

impl Drop for ExternalCheckerContext {
    fn drop(&mut self) {
        // Ensure any Python nodes retained past the dispatch observe an invalid store rather
        // than dereferencing freed AST data.
        self.store.invalidate();
    }
}

fn node_ids_to_u32(ids: impl Iterator<Item = NodeId>) -> Vec<u32> {
    ids.map(NodeId::as_u32).collect()
}

fn build_context(
    py: Python<'_>,
    module_types: &ModuleTypes,
    config: &PyObject,
    rule: &crate::external::ast::rule::ExternalAstRule,
    range: TextRange,
    checker: &Checker<'_>,
) -> PyResult<RuntimeContext> {
    let reporter = PyReporter::new(py, rule, range)?;
    let semantic = SemanticModelView::new(py, checker, module_types.projection)?;
    let context = module_types
        .context
        .bind(py)
        .call1((
            rule.code.as_str(),
            rule.name.as_str(),
            config.clone_ref(py),
            reporter.clone_ref(py),
            semantic.clone_ref(py),
        ))?
        .into();

    Ok(RuntimeContext {
        context,
        reporter,
        semantic,
    })
}

fn config_to_python(
    py: Python<'_>,
    config: Option<&crate::external::ast::rule::ExternalLinterConfig>,
) -> PyResult<PyObject> {
    match config {
        Some(config) => table_to_python(py, config.data()),
        None => Ok(PyDict::new(py).into()),
    }
}

fn table_to_python(py: Python<'_>, table: &TomlTable) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in table {
        dict.set_item(key, value_to_python(py, value)?)?;
    }
    Ok(dict.into())
}

fn value_to_python(py: Python<'_>, value: &TomlValue) -> PyResult<PyObject> {
    match value {
        TomlValue::Boolean(v) => {
            let obj = (*v).into_pyobject(py)?;
            Ok((*obj).clone().into())
        }
        TomlValue::Integer(v) => {
            let obj = (*v).into_pyobject(py)?;
            Ok((*obj).clone().into())
        }
        TomlValue::Float(v) => {
            let obj = (*v).into_pyobject(py)?;
            Ok((*obj).clone().into())
        }
        TomlValue::String(v) => {
            let obj = v.clone().into_pyobject(py)?;
            Ok((*obj).clone().into())
        }
        TomlValue::Datetime(v) => {
            let obj = v.to_string().into_pyobject(py)?;
            Ok((*obj).clone().into())
        }
        TomlValue::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_python(py, item)?)?;
            }
            Ok(list.into())
        }
        TomlValue::Table(table) => table_to_python(py, table),
    }
}

fn rule_applicable_to_expr(
    rule: &crate::external::ast::rule::ExternalAstRule,
    kind: ExprKind,
    expr: &Expr,
    call_callee_cache: &mut CachedCallee,
) -> bool {
    match rule.call_callee() {
        Some(matcher) => {
            if kind != ExprKind::Call {
                return false;
            }

            let callee = call_callee_cache.resolve(expr);
            match callee {
                Some(callee) => matcher.regex().is_match(callee),
                None => false,
            }
        }
        None => true,
    }
}

fn extract_call_callee(expr: &Expr) -> Option<String> {
    let call = expr.as_call_expr()?;
    UnqualifiedName::from_expr(call.func.as_ref()).map(|name| name.to_string())
}

#[derive(Default)]
struct CachedCallee {
    cached: CachedValue,
}

#[derive(Default)]
enum CachedValue {
    #[default]
    Unknown,
    Known(Option<String>),
}

impl CachedCallee {
    fn resolve<'expr>(&'expr mut self, expr: &Expr) -> Option<&'expr str> {
        if matches!(self.cached, CachedValue::Unknown) {
            self.cached = CachedValue::Known(extract_call_callee(expr));
        }

        match self.cached {
            CachedValue::Known(Some(ref value)) => Some(value.as_str()),
            _ => None,
        }
    }
}

fn ensure_registry_verified(id: u64, registry: &ExternalLintRegistry) {
    let cache = VERIFIED_REGISTRIES.get_or_init(|| Mutex::new(FxHashSet::default()));
    let mut cache = cache.lock().expect("verification cache poisoned");
    if cache.contains(&id) {
        return;
    }
    verify_registry_scripts(registry)
        .unwrap_or_else(|error| panic!("failed to compile external scripts: {error}"));
    cache.insert(id);
}

pub fn verify_registry_scripts(registry: &ExternalLintRegistry) -> Result<(), ExternalLinterError> {
    compile_scripts(registry).map(|_| ())
}

mod reporter {
    #![allow(unsafe_op_in_unsafe_fn)]

    use crate::checkers::ast::Checker;
    use crate::rules::ruff::rules::external_ast::ExternalLinter as ExternalLinterViolation;
    use pyo3::prelude::*;
    use ruff_db::diagnostic::SecondaryCode;
    use ruff_text_size::{TextRange, TextSize};
    use std::cell::RefCell;

    #[pyclass(module = "ruff_external", unsendable)]
    pub(crate) struct PyReporter {
        diagnostics: RefCell<Vec<PendingDiagnostic>>,
        rule_code: String,
        rule_name: String,
        default_span: (u32, u32),
    }

    #[derive(Debug)]
    struct PendingDiagnostic {
        message: String,
        span: Option<(u32, u32)>,
    }

    impl PyReporter {
        pub(crate) fn new(
            py: Python<'_>,
            rule: &crate::external::ast::rule::ExternalAstRule,
            range: TextRange,
        ) -> PyResult<Py<PyReporter>> {
            Py::new(
                py,
                PyReporter {
                    diagnostics: RefCell::new(Vec::new()),
                    rule_code: rule.code.as_str().to_string(),
                    rule_name: rule.name.clone(),
                    default_span: (range.start().to_u32(), range.end().to_u32()),
                },
            )
        }

        pub(crate) fn drain_into(&self, checker: &Checker<'_>) {
            let pending = std::mem::take(&mut *self.diagnostics.borrow_mut());
            for diagnostic in pending {
                let range = self.resolve_span(diagnostic.span);
                let mut emitted = checker.report_diagnostic(
                    ExternalLinterViolation {
                        rule_name: self.rule_name.clone(),
                        message: diagnostic.message,
                    },
                    range,
                );
                emitted.set_secondary_code(SecondaryCode::new(self.rule_code.clone()));
            }
        }

        fn resolve_span(&self, span: Option<(u32, u32)>) -> TextRange {
            let (start, end) = match span {
                Some((start, end)) if end >= start => (start, end),
                _ => self.default_span,
            };
            TextRange::new(TextSize::new(start), TextSize::new(end))
        }
    }

    #[pymethods]
    impl PyReporter {
        #[pyo3(signature = (message, span=None))]
        fn __call__(&self, message: &str, span: Option<(u32, u32)>) {
            self.diagnostics.borrow_mut().push(PendingDiagnostic {
                message: message.to_string(),
                span,
            });
        }
    }
}

use reporter::PyReporter;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    use crate::external::ast::rule::{
        ExternalAstLinter, ExternalAstRule, ExternalRuleCode, ExternalRuleScript,
    };
    use crate::external::ast::target::{AstTarget, StmtKind};
    use crate::external::{PyprojectExternalLinterEntry, load_linter_from_entry};
    use crate::registry::Rule;
    use crate::settings::LinterSettings;
    use crate::source_kind::SourceKind;
    use crate::test::test_contents;
    use tempfile::tempdir;

    fn basic_rule(code: &str, script: ExternalRuleScript) -> ExternalAstRule {
        ExternalAstRule::new(
            ExternalRuleCode::new(code).unwrap(),
            "ExampleRule",
            None::<&str>,
            vec![AstTarget::Stmt(StmtKind::FunctionDef)],
            script,
            None,
        )
    }

    fn write_fixture(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn interpreter_runs_basic_code() {
        let runtime = ExternalLintRuntime::new(ExternalLintRegistry::new());
        runtime
            .runtime_cache
            .with_runtime(runtime.registry(), |_, _| {
                with_attached_python(|py| {
                    let code = CString::new("40 + 2").unwrap();
                    let result: i32 = py
                        .eval(code.as_c_str(), None, None)
                        .unwrap()
                        .extract()
                        .unwrap();
                    assert_eq!(result, 42);
                });
            });
    }

    #[test]
    fn compile_errors_surface_during_validation() {
        let mut registry = ExternalLintRegistry::new();
        let rule = basic_rule(
            "EXT001",
            ExternalRuleScript::file(PathBuf::from("broken.py"), "def broken(:\n"),
        );
        let linter = ExternalAstLinter::new("broken", "Broken", None::<&str>, true, vec![rule]);
        registry.insert_linter(linter).unwrap();

        let other_rule = basic_rule(
            "EXT002",
            ExternalRuleScript::file(PathBuf::from("other.py"), "def also_broken(:\n"),
        );
        let other_linter =
            ExternalAstLinter::new("other", "Other", None::<&str>, true, vec![other_rule]);
        registry.insert_linter(other_linter).unwrap();

        let err = verify_registry_scripts(&registry).expect_err("expected compile failure");
        match err {
            ExternalLinterError::ScriptCompile { message } => {
                assert!(message.contains("broken"), "message: {message}");
                assert!(message.contains("other"), "message: {message}");
                assert!(
                    message.lines().count() >= 2,
                    "expected multiple lines: {message}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn current_statement_and_scope_instance_types() {
        let script = r#"
import ruff_external

def check_stmt(node, ctx):
    stmt = ctx.semantic.current_statement()
    scope = ctx.semantic.current_scope()
    ctx.report(
        f"stmt_kind={stmt._kind};"
        f"stmt_is_node={isinstance(stmt, ruff_external.Node)};"
        f"scope_kind={scope.kind};"
        f"scope_is_handle={isinstance(scope, ruff_external.ScopeHandle)}"
    )
"#;

        let temp = tempdir().unwrap();
        let linter_path = temp.path().join("linter.toml");
        let script_path = temp.path().join("scope_shape.py");

        write_fixture(&script_path, script);
        write_fixture(
            &linter_path,
            r#"
name = "ScopeShape"

[[rule]]
code = "EXT001"
name = "ScopeShape"
targets = ["stmt:If"]
script = "scope_shape.py"
"#,
        );

        let entry = PyprojectExternalLinterEntry {
            toml_path: linter_path,
            enabled: true,
        };
        let linter = load_linter_from_entry("ext", &entry).unwrap();
        let mut registry = ExternalLintRegistry::new();
        registry.insert_linter(linter).unwrap();

        let mut settings = LinterSettings::default();
        settings.rules.enable(Rule::ExternalLinter, false);
        settings.external_ast = Some(registry);

        let source = "def outer(x):\n    if x:\n        return x\n";
        let diagnostics = test_contents(
            &SourceKind::Python {
                code: source.into(),
                is_stub: false,
            },
            Path::new("test.py"),
            &settings,
        )
        .0;

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].primary_message(),
            "ScopeShape: stmt_kind=If;stmt_is_node=True;scope_kind=function;scope_is_handle=True"
        );
    }

    #[test]
    fn semantic_visit_dispatches_callbacks() {
        let script = r#"
def check_stmt(node, ctx):
    calls = []
    ctx.semantic.visit(node, expr_Call=lambda n: calls.append(n))

    calls_skipped = []
    ctx.semantic.visit(
        node,
        stmt_If=lambda n: False,
        expr_Call=lambda n: calls_skipped.append(n),
    )

    ctx.report(f"calls={len(calls)};calls_skipped={len(calls_skipped)}")
"#;

        let temp = tempdir().unwrap();
        let linter_path = temp.path().join("linter.toml");
        let script_path = temp.path().join("visit.py");

        write_fixture(&script_path, script);
        write_fixture(
            &linter_path,
            r#"
name = "Visit"

[[rule]]
code = "EXT001"
name = "Visit"
targets = ["stmt:If"]
script = "visit.py"
"#,
        );

        let entry = PyprojectExternalLinterEntry {
            toml_path: linter_path,
            enabled: true,
        };
        let linter = load_linter_from_entry("ext", &entry).unwrap();
        let mut registry = ExternalLintRegistry::new();
        registry.insert_linter(linter).unwrap();

        let mut settings = LinterSettings::default();
        settings.rules.enable(Rule::ExternalLinter, false);
        settings.external_ast = Some(registry);

        let source = "if True:\n    print('a')\n    print('b')\n";
        let diagnostics = test_contents(
            &SourceKind::Python {
                code: source.into(),
                is_stub: false,
            },
            Path::new("test.py"),
            &settings,
        )
        .0;

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].primary_message(),
            "Visit: calls=2;calls_skipped=0"
        );
    }

    #[test]
    fn semantic_resolve_qualified_name_is_none_when_node_has_no_semantic_id() {
        // `resolve_qualified_name` returns `Optional[str]`; it should not raise when invoked on an
        // AST node that isn't indexed by the semantic model (for example, nodes in typing-only
        // contexts like annotations).
        let script = r#"
def check_stmt(node, ctx):
    qualified = ctx.semantic.resolve_qualified_name(node.annotation)
    ctx.report(f"qualified={qualified}")
"#;

        let temp = tempdir().unwrap();
        let linter_path = temp.path().join("linter.toml");
        let script_path = temp.path().join("resolve_qualified_name.py");

        write_fixture(&script_path, script);
        write_fixture(
            &linter_path,
            r#"
name = "ResolveQualifiedName"

[[rule]]
code = "EXT001"
name = "ResolveQualifiedName"
targets = ["stmt:AnnAssign"]
script = "resolve_qualified_name.py"
"#,
        );

        let entry = PyprojectExternalLinterEntry {
            toml_path: linter_path,
            enabled: true,
        };
        let linter = load_linter_from_entry("ext", &entry).unwrap();
        let mut registry = ExternalLintRegistry::new();
        registry.insert_linter(linter).unwrap();

        let mut settings = LinterSettings::default();
        settings.rules.enable(Rule::ExternalLinter, false);
        settings.external_ast = Some(registry);

        let source = "x: int | None = None\n";
        let diagnostics = test_contents(
            &SourceKind::Python {
                code: source.into(),
                is_stub: false,
            },
            Path::new("test.py"),
            &settings,
        )
        .0;

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].primary_message(),
            "ResolveQualifiedName: qualified=None"
        );
    }

    #[test]
    fn current_scope_chain_matches_parent_statement() {
        let script = r#"
import ruff_external

def check_stmt(node, ctx):
    scopes = ctx.semantic.current_scope_ids()
    handles = ctx.semantic.current_scopes()
    parent_if = ctx.semantic.parent_statement(node)
    parent_func = ctx.semantic.parent_statement(parent_if) if parent_if else None
    kinds = [h.kind for h in handles]
    ctx.report(
        f"scopes_len={len(scopes)};"
        f"handles_match={[h.id for h in handles] == scopes};"
        f"scope_kinds={kinds};"
        f"parent_func_kind={parent_func['_kind'] if parent_func else None};"
        f"parent_id_chain={handles[0].parent_id == handles[1].id if len(handles) > 1 else None}"
    )
"#;

        let temp = tempdir().unwrap();
        let linter_path = temp.path().join("linter.toml");
        let script_path = temp.path().join("scope_chain.py");

        write_fixture(&script_path, script);
        write_fixture(
            &linter_path,
            r#"
name = "ScopeChain"

[[rule]]
code = "EXT002"
name = "ScopeChain"
targets = ["stmt:Return"]
script = "scope_chain.py"
"#,
        );

        let entry = PyprojectExternalLinterEntry {
            toml_path: linter_path,
            enabled: true,
        };
        let linter = load_linter_from_entry("ext", &entry).unwrap();
        let mut registry = ExternalLintRegistry::new();
        registry.insert_linter(linter).unwrap();

        let mut settings = LinterSettings::default();
        settings.rules.enable(Rule::ExternalLinter, false);
        settings.external_ast = Some(registry);

        let source = r#"
def outer(a):
    def inner(b):
        if b:
            return b
    return inner(a)
"#;
        let diagnostics = test_contents(
            &SourceKind::Python {
                code: source.into(),
                is_stub: false,
            },
            Path::new("test.py"),
            &settings,
        )
        .0;

        assert!(!diagnostics.is_empty());
        let messages: Vec<_> = diagnostics
            .iter()
            .map(ruff_db::diagnostic::Diagnostic::primary_message)
            .collect();
        let with_parent = messages
            .iter()
            .find(|message| message.contains("parent_func_kind=FunctionDef"))
            .expect("missing diagnostic for inner function return");
        assert!(with_parent.contains("handles_match=True"));
        assert!(with_parent.contains("parent_id_chain=True"));
        assert!(with_parent.contains("scope_kinds=['function'"));
    }

    #[test]
    fn function_def_deferred_runs_after_body() {
        let script = r#"
import ruff_external

def check_stmt(node, ctx):
    first_stmt = node.body[0]
    # Access semantic information that requires node ids from the body.
    ctx.semantic.parent_statement(first_stmt)
    ctx.report("deferred-ok", node._span)
"#;

        let temp = tempdir().unwrap();
        let linter_path = temp.path().join("linter.toml");
        let script_path = temp.path().join("function_deferred.py");

        write_fixture(&script_path, script);
        write_fixture(
            &linter_path,
            r#"
name = "FunctionDeferred"

[[rule]]
code = "EXT003"
name = "FunctionDeferred"
targets = ["stmt:FunctionDefDeferred"]
script = "function_deferred.py"
"#,
        );

        let entry = PyprojectExternalLinterEntry {
            toml_path: linter_path,
            enabled: true,
        };
        let linter = load_linter_from_entry("ext", &entry).unwrap();
        let mut registry = ExternalLintRegistry::new();
        registry.insert_linter(linter).unwrap();

        let mut settings = LinterSettings::default();
        settings.rules.enable(Rule::ExternalLinter, false);
        settings.external_ast = Some(registry);

        let source = r#"
def outer():
    value = 1
    return value
"#;
        let diagnostics = test_contents(
            &SourceKind::Python {
                code: source.into(),
                is_stub: false,
            },
            Path::new("test.py"),
            &settings,
        )
        .0;

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].primary_message(),
            "FunctionDeferred: deferred-ok"
        );
    }
}

pub(crate) mod semantic {
    use super::{
        Bound, ProjectionTypesRef, Py, PyAny, PyObject, PyResult, Python, QualifiedName, Ranged,
        expr_to_python, node_ids_to_u32, pyclass, pymethods, stmt_to_python,
    };
    use crate::checkers::ast::Checker;
    use crate::external::ast::python::node_handle;
    use crate::external::ast::python::node_to_python;
    use crate::external::ast::target::{AstTarget, ExprKind, StmtKind};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::types::{PyAnyMethods, PyBool, PyDict, PyDictMethods};
    use ruff_python_ast::visitor::source_order::{self, SourceOrderVisitor, TraversalSignal};
    use ruff_python_ast::{AnyNodeRef, NodeKind};
    use ruff_python_ast::{Expr, ExprContext, HasNodeIndex};
    use ruff_python_semantic::{
        BindingId, BindingKind, ExecutionContext, GeneratorKind, ImportedName, Modules, NodeId,
        NodeRef, ResolvedReference, ResolvedReferenceId, Scope, ScopeId, ScopeKind,
    };
    use rustc_hash::FxHashMap;
    use std::{cell::Cell, ptr};

    #[pyclass(module = "ruff_external", name = "SemanticModel", unsendable)]
    pub(crate) struct SemanticModelView {
        state: SemanticState,
        projection: ProjectionTypesRef,
    }

    impl SemanticModelView {
        pub(crate) fn new(
            py: Python<'_>,
            checker: &Checker<'_>,
            projection: ProjectionTypesRef,
        ) -> PyResult<Py<Self>> {
            Py::new(
                py,
                SemanticModelView {
                    state: SemanticState::new(checker),
                    projection,
                },
            )
        }

        pub(crate) fn deactivate(&self) {
            self.state.deactivate();
        }

        fn with_checker<R>(&self, f: impl FnOnce(&Checker<'_>) -> PyResult<R>) -> PyResult<R> {
            self.state.with_checker(f)
        }
    }

    #[pymethods]
    impl SemanticModelView {
        fn resolve_name(&self, node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(None);
                };
                Ok(resolve_name_string(checker, node_id))
            })
        }

        #[pyo3(signature = (node, /, **handlers))]
        fn visit(
            &self,
            node: &Bound<'_, PyAny>,
            handlers: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<()> {
            let py = node.py();

            let mut callbacks: FxHashMap<NodeKind, Py<PyAny>> = FxHashMap::default();
            if let Some(handlers) = handlers {
                for (key, value) in handlers.iter() {
                    let key: &str = key.extract()?;
                    let spec = if let Some(kind) = key.strip_prefix("stmt_") {
                        format!("stmt:{kind}")
                    } else if let Some(kind) = key.strip_prefix("expr_") {
                        format!("expr:{kind}")
                    } else {
                        key.to_string()
                    };

                    let target = spec
                        .parse::<AstTarget>()
                        .map_err(|err| PyValueError::new_err(err.to_string()))?;
                    let target = normalize_deferred_target(target);

                    let kind = node_kind_for_target(target);
                    callbacks.insert(kind, value.unbind());
                }
            }

            if callbacks.is_empty() {
                return Ok(());
            }

            self.with_checker(|checker| {
                let node_id = checked_node_id(node, checker)?;
                let semantic = checker.semantic();
                let root = *semantic.node(node_id);

                let mut visitor = PythonCallbackVisitor {
                    py,
                    locator: checker.locator(),
                    projection: self.projection,
                    callbacks,
                    error: None,
                };

                match root {
                    NodeRef::Stmt(stmt) => source_order::walk_stmt(&mut visitor, stmt),
                    NodeRef::Expr(expr) => source_order::walk_expr(&mut visitor, expr),
                }

                if let Some(err) = visitor.error {
                    Err(err)
                } else {
                    Ok(())
                }
            })
        }

        fn resolve_binding(&self, node: &Bound<'_, PyAny>) -> PyResult<Option<PyBinding>> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(None);
                };
                Ok(resolve_name_binding(checker, node_id))
            })
        }

        fn only_binding(&self, node: &Bound<'_, PyAny>) -> PyResult<Option<PyBinding>> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(None);
                };
                let semantic = checker.semantic();
                let binding = match semantic.node(node_id).as_expression() {
                    Some(Expr::Name(name)) => semantic.only_binding(name),
                    _ => None,
                };
                Ok(binding.map(|binding_id| PyBinding::new(checker, binding_id)))
            })
        }

        fn resolve_qualified_name(&self, node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(None);
                };
                let semantic = checker.semantic();
                let qualified = semantic
                    .node(node_id)
                    .as_expression()
                    .and_then(|expr| semantic.resolve_qualified_name(expr))
                    .map(|value| value.to_string());
                Ok(qualified)
            })
        }

        fn lookup_attribute(&self, node: &Bound<'_, PyAny>) -> PyResult<Option<PyBinding>> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(None);
                };
                let semantic = checker.semantic();
                let binding = semantic
                    .node(node_id)
                    .as_expression()
                    .and_then(|expr| semantic.lookup_attribute(expr));
                Ok(binding.map(|binding_id| PyBinding::new(checker, binding_id)))
            })
        }

        fn resolve_qualified_import_name(
            &self,
            module: &str,
            member: &str,
        ) -> PyResult<Option<String>> {
            self.with_checker(|checker| {
                Ok(checker
                    .semantic()
                    .resolve_qualified_import_name(module, member)
                    .map(ImportedName::into_name))
            })
        }

        fn current_statement(&self, py: Python<'_>) -> PyResult<PyObject> {
            self.with_checker(|checker| {
                stmt_to_python(
                    py,
                    checker.locator(),
                    checker.semantic().current_statement(),
                    self.projection,
                )
            })
        }

        fn current_statement_parent(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
            self.with_checker(|checker| {
                checker
                    .semantic()
                    .current_statement_parent()
                    .map(|stmt| stmt_to_python(py, checker.locator(), stmt, self.projection))
                    .transpose()
            })
        }

        fn current_statements(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
            self.with_checker(|checker| {
                checker
                    .semantic()
                    .current_statements()
                    .map(|stmt| stmt_to_python(py, checker.locator(), stmt, self.projection))
                    .collect()
            })
        }

        fn current_statement_id(&self) -> PyResult<Option<u32>> {
            self.with_checker(|checker| {
                Ok(checker
                    .semantic()
                    .current_statement_id()
                    .map(NodeId::as_u32))
            })
        }

        fn current_statement_parent_id(&self) -> PyResult<Option<u32>> {
            self.with_checker(|checker| {
                Ok(checker
                    .semantic()
                    .current_statement_parent_id()
                    .map(NodeId::as_u32))
            })
        }

        fn current_statement_ids(&self) -> PyResult<Vec<u32>> {
            self.with_checker(|checker| {
                Ok(node_ids_to_u32(checker.semantic().current_statement_ids()))
            })
        }

        fn parent_statement(
            &self,
            node: &Bound<'_, PyAny>,
            py: Python<'_>,
        ) -> PyResult<Option<PyObject>> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(None);
                };
                checker
                    .semantic()
                    .parent_statement(node_id)
                    .map(|stmt| stmt_to_python(py, checker.locator(), stmt, self.projection))
                    .transpose()
            })
        }

        fn parent_statement_id(&self, node: &Bound<'_, PyAny>) -> PyResult<Option<u32>> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(None);
                };
                Ok(checker
                    .semantic()
                    .parent_statement_id(node_id)
                    .map(NodeId::as_u32))
            })
        }

        fn same_branch(&self, left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<bool> {
            self.with_checker(|checker| {
                let left_id = checked_node_id(left, checker)?;
                let right_id = checked_node_id(right, checker)?;
                Ok(checker.semantic().same_branch(left_id, right_id))
            })
        }

        fn dominates(&self, left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<bool> {
            self.with_checker(|checker| {
                let left_id = checked_node_id(left, checker)?;
                let right_id = checked_node_id(right, checker)?;
                Ok(checker.semantic().dominates(left_id, right_id))
            })
        }

        fn current_expression(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
            self.with_checker(|checker| {
                checker
                    .semantic()
                    .current_expression()
                    .map(|expr| expr_to_python(py, checker.locator(), expr, self.projection))
                    .transpose()
            })
        }

        fn current_expression_parent(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
            self.with_checker(|checker| {
                checker
                    .semantic()
                    .current_expression_parent()
                    .map(|expr| expr_to_python(py, checker.locator(), expr, self.projection))
                    .transpose()
            })
        }

        fn current_expression_grandparent(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
            self.with_checker(|checker| {
                checker
                    .semantic()
                    .current_expression_grandparent()
                    .map(|expr| expr_to_python(py, checker.locator(), expr, self.projection))
                    .transpose()
            })
        }

        fn current_expressions(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
            self.with_checker(|checker| {
                checker
                    .semantic()
                    .current_expressions()
                    .map(|expr| expr_to_python(py, checker.locator(), expr, self.projection))
                    .collect()
            })
        }

        fn global_declaration(&self, name: &str) -> PyResult<Option<(u32, u32)>> {
            if name.is_empty() {
                return Err(PyValueError::new_err("name must be non-empty"));
            }
            self.with_checker(|checker| {
                Ok(checker
                    .semantic()
                    .global(name)
                    .map(|range| (range.start().to_u32(), range.end().to_u32())))
            })
        }

        fn resolve_nonlocal(&self, name: &str) -> PyResult<Option<(PyScopeHandle, PyBinding)>> {
            if name.is_empty() {
                return Err(PyValueError::new_err("name must be non-empty"));
            }
            self.with_checker(|checker| {
                let semantic = checker.semantic();
                Ok(semantic.nonlocal(name).map(|(scope_id, binding_id)| {
                    (
                        PyScopeHandle::new(scope_id, &semantic.scopes[scope_id]),
                        PyBinding::new(checker, binding_id),
                    )
                }))
            })
        }

        fn shadowed_binding(&self, node: &Bound<'_, PyAny>) -> PyResult<Option<PyBinding>> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(None);
                };
                let Some(binding_id) = resolve_binding_id(checker, node_id) else {
                    return Ok(None);
                };
                Ok(checker
                    .semantic()
                    .shadowed_binding(binding_id)
                    .map(|shadowed_id| PyBinding::new(checker, shadowed_id)))
            })
        }

        fn shadowed_bindings(
            &self,
            node: &Bound<'_, PyAny>,
        ) -> PyResult<Vec<(PyBinding, PyBinding, bool)>> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(Vec::new());
                };
                let Some(binding_id) = resolve_binding_id(checker, node_id) else {
                    return Ok(Vec::new());
                };
                let semantic = checker.semantic();
                let scope_id = semantic.binding(binding_id).scope;
                Ok(semantic
                    .shadowed_bindings(scope_id, binding_id)
                    .map(|shadow| {
                        (
                            PyBinding::new(checker, shadow.binding_id()),
                            PyBinding::new(checker, shadow.shadowed_id()),
                            shadow.same_scope(),
                        )
                    })
                    .collect())
            })
        }

        fn at_top_level(&self) -> PyResult<bool> {
            self.with_checker(|checker| Ok(checker.semantic().at_top_level()))
        }

        fn in_async_context(&self) -> PyResult<bool> {
            self.with_checker(|checker| Ok(checker.semantic().in_async_context()))
        }

        fn in_nested_union(&self) -> PyResult<bool> {
            self.with_checker(|checker| Ok(checker.semantic().in_nested_union()))
        }

        fn inside_optional(&self) -> PyResult<bool> {
            self.with_checker(|checker| Ok(checker.semantic().inside_optional()))
        }

        fn in_nested_literal(&self) -> PyResult<bool> {
            self.with_checker(|checker| Ok(checker.semantic().in_nested_literal()))
        }

        fn current_scope(&self) -> PyResult<PyScopeHandle> {
            self.with_checker(|checker| {
                let semantic = checker.semantic();
                Ok(PyScopeHandle::new(
                    semantic.scope_id,
                    semantic.current_scope(),
                ))
            })
        }

        fn current_scope_id(&self) -> PyResult<u32> {
            self.with_checker(|checker| Ok(checker.semantic().scope_id.as_u32()))
        }

        fn current_scope_ids(&self) -> PyResult<Vec<u32>> {
            self.with_checker(|checker| {
                Ok(checker
                    .semantic()
                    .current_scope_ids()
                    .map(ScopeId::as_u32)
                    .collect())
            })
        }

        fn current_scopes(&self) -> PyResult<Vec<PyScopeHandle>> {
            self.with_checker(|checker| {
                let semantic = checker.semantic();
                Ok(semantic
                    .current_scope_ids()
                    .map(|scope_id| PyScopeHandle::new(scope_id, &semantic.scopes[scope_id]))
                    .collect())
            })
        }

        fn first_non_type_parent_scope(&self) -> PyResult<Option<PyScopeHandle>> {
            self.with_checker(|checker| {
                let semantic = checker.semantic();
                let parent_id = semantic.first_non_type_parent_scope_id(semantic.scope_id);
                Ok(parent_id
                    .map(|scope_id| PyScopeHandle::new(scope_id, &semantic.scopes[scope_id])))
            })
        }

        fn first_non_type_parent_scope_id(&self) -> PyResult<Option<u32>> {
            self.with_checker(|checker| {
                let semantic = checker.semantic();
                Ok(semantic
                    .first_non_type_parent_scope_id(semantic.scope_id)
                    .map(ScopeId::as_u32))
            })
        }

        fn match_builtin_expr(&self, node: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(false);
                };
                Ok(match checker.semantic().node(node_id).as_expression() {
                    Some(expr) => checker.semantic().match_builtin_expr(expr, name),
                    None => false,
                })
            })
        }

        fn resolve_builtin_symbol(&self, node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(None);
                };
                Ok(checker
                    .semantic()
                    .node(node_id)
                    .as_expression()
                    .and_then(|expr| checker.semantic().resolve_builtin_symbol(expr))
                    .map(ToString::to_string))
            })
        }

        fn match_typing_expr(&self, node: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
            self.with_checker(|checker| {
                let Some(node_id) = maybe_node_id(node, checker)? else {
                    return Ok(false);
                };
                Ok(match checker.semantic().node(node_id).as_expression() {
                    Some(expr) => checker.semantic().match_typing_expr(expr, name),
                    None => false,
                })
            })
        }

        fn match_typing_qualified_name(
            &self,
            qualified_name: &str,
            target: &str,
        ) -> PyResult<bool> {
            if qualified_name.is_empty() || target.is_empty() {
                return Err(PyValueError::new_err(
                    "qualified_name and target must be non-empty",
                ));
            }
            self.with_checker(|checker| {
                let qualified = QualifiedName::from_dotted_name(qualified_name);
                Ok(checker
                    .semantic()
                    .match_typing_qualified_name(&qualified, target))
            })
        }

        fn seen_typing(&self) -> PyResult<bool> {
            self.with_checker(|checker| Ok(checker.semantic().seen_typing()))
        }

        fn in_annotation(&self) -> PyResult<bool> {
            self.with_checker(|checker| Ok(checker.semantic().in_annotation()))
        }

        fn in_forward_reference(&self) -> PyResult<bool> {
            self.with_checker(|checker| Ok(checker.semantic().in_forward_reference()))
        }

        fn seen_module(&self, name: &str) -> PyResult<bool> {
            self.with_checker(|checker| {
                Ok(module_from_name(name)
                    .map(|module| checker.semantic().seen_module(module))
                    .unwrap_or(false))
            })
        }

        fn has_builtin_binding(&self, name: &str) -> PyResult<bool> {
            self.with_checker(|checker| Ok(checker.semantic().has_builtin_binding(name)))
        }

        fn execution_context(&self) -> PyResult<&'static str> {
            self.with_checker(|checker| {
                Ok(match checker.semantic().execution_context() {
                    ExecutionContext::Runtime => "runtime",
                    ExecutionContext::Typing => "typing",
                })
            })
        }

        fn in_typing_only_annotation(&self) -> PyResult<bool> {
            self.with_checker(|checker| Ok(checker.semantic().in_typing_only_annotation()))
        }

        fn __repr__(&self) -> String {
            let active = self.state.is_active();
            format!("SemanticModel(active={active})")
        }
    }

    fn normalize_deferred_target(target: AstTarget) -> AstTarget {
        match target {
            AstTarget::Stmt(StmtKind::FunctionDefDeferred) => {
                AstTarget::Stmt(StmtKind::FunctionDef)
            }
            AstTarget::Expr(ExprKind::LambdaDeferred) => AstTarget::Expr(ExprKind::Lambda),
            target => target,
        }
    }

    fn node_kind_for_target(target: AstTarget) -> NodeKind {
        match target {
            AstTarget::Stmt(kind) => match kind {
                StmtKind::FunctionDef | StmtKind::FunctionDefDeferred => NodeKind::StmtFunctionDef,
                StmtKind::ClassDef => NodeKind::StmtClassDef,
                StmtKind::Return => NodeKind::StmtReturn,
                StmtKind::Delete => NodeKind::StmtDelete,
                StmtKind::TypeAlias => NodeKind::StmtTypeAlias,
                StmtKind::Assign => NodeKind::StmtAssign,
                StmtKind::AugAssign => NodeKind::StmtAugAssign,
                StmtKind::AnnAssign => NodeKind::StmtAnnAssign,
                StmtKind::For => NodeKind::StmtFor,
                StmtKind::While => NodeKind::StmtWhile,
                StmtKind::If => NodeKind::StmtIf,
                StmtKind::With => NodeKind::StmtWith,
                StmtKind::Match => NodeKind::StmtMatch,
                StmtKind::Raise => NodeKind::StmtRaise,
                StmtKind::Try => NodeKind::StmtTry,
                StmtKind::Assert => NodeKind::StmtAssert,
                StmtKind::Import => NodeKind::StmtImport,
                StmtKind::ImportFrom => NodeKind::StmtImportFrom,
                StmtKind::Global => NodeKind::StmtGlobal,
                StmtKind::Nonlocal => NodeKind::StmtNonlocal,
                StmtKind::Expr => NodeKind::StmtExpr,
                StmtKind::Pass => NodeKind::StmtPass,
                StmtKind::Break => NodeKind::StmtBreak,
                StmtKind::Continue => NodeKind::StmtContinue,
                StmtKind::IpyEscapeCommand => NodeKind::StmtIpyEscapeCommand,
            },
            AstTarget::Expr(kind) => match kind {
                ExprKind::Attribute => NodeKind::ExprAttribute,
                ExprKind::Await => NodeKind::ExprAwait,
                ExprKind::BinOp => NodeKind::ExprBinOp,
                ExprKind::BoolOp => NodeKind::ExprBoolOp,
                ExprKind::BooleanLiteral => NodeKind::ExprBooleanLiteral,
                ExprKind::BytesLiteral => NodeKind::ExprBytesLiteral,
                ExprKind::Call => NodeKind::ExprCall,
                ExprKind::Compare => NodeKind::ExprCompare,
                ExprKind::Dict => NodeKind::ExprDict,
                ExprKind::DictComp => NodeKind::ExprDictComp,
                ExprKind::EllipsisLiteral => NodeKind::ExprEllipsisLiteral,
                ExprKind::FString => NodeKind::ExprFString,
                ExprKind::TString => NodeKind::ExprTString,
                ExprKind::Generator => NodeKind::ExprGenerator,
                ExprKind::If => NodeKind::ExprIf,
                ExprKind::IpyEscapeCommand => NodeKind::ExprIpyEscapeCommand,
                ExprKind::Lambda | ExprKind::LambdaDeferred => NodeKind::ExprLambda,
                ExprKind::List => NodeKind::ExprList,
                ExprKind::ListComp => NodeKind::ExprListComp,
                ExprKind::Name => NodeKind::ExprName,
                ExprKind::Named => NodeKind::ExprNamed,
                ExprKind::NoneLiteral => NodeKind::ExprNoneLiteral,
                ExprKind::NumberLiteral => NodeKind::ExprNumberLiteral,
                ExprKind::Set => NodeKind::ExprSet,
                ExprKind::SetComp => NodeKind::ExprSetComp,
                ExprKind::Slice => NodeKind::ExprSlice,
                ExprKind::Starred => NodeKind::ExprStarred,
                ExprKind::StringLiteral => NodeKind::ExprStringLiteral,
                ExprKind::Subscript => NodeKind::ExprSubscript,
                ExprKind::Tuple => NodeKind::ExprTuple,
                ExprKind::UnaryOp => NodeKind::ExprUnaryOp,
                ExprKind::Yield => NodeKind::ExprYield,
                ExprKind::YieldFrom => NodeKind::ExprYieldFrom,
            },
        }
    }

    struct PythonCallbackVisitor<'py, 'a> {
        py: Python<'py>,
        locator: &'a crate::Locator<'a>,
        projection: ProjectionTypesRef,
        callbacks: FxHashMap<NodeKind, Py<PyAny>>,
        error: Option<pyo3::PyErr>,
    }

    impl<'a> SourceOrderVisitor<'a> for PythonCallbackVisitor<'_, 'a> {
        fn enter_node(&mut self, node: AnyNodeRef<'a>) -> TraversalSignal {
            if self.error.is_some() {
                return TraversalSignal::Skip;
            }

            let Some(callback) = self.callbacks.get(&node.kind()) else {
                return TraversalSignal::Traverse;
            };

            let projected = match node_to_python(self.py, self.locator, node, self.projection) {
                Ok(value) => value,
                Err(err) => {
                    self.error = Some(err);
                    return TraversalSignal::Skip;
                }
            };

            let result = match callback.bind(self.py).call1((projected,)) {
                Ok(result) => result,
                Err(err) => {
                    self.error = Some(err);
                    return TraversalSignal::Skip;
                }
            };

            if result.is_instance_of::<PyBool>() {
                match result.extract::<bool>() {
                    Ok(true) => TraversalSignal::Traverse,
                    Ok(false) => TraversalSignal::Skip,
                    Err(err) => {
                        self.error = Some(err);
                        TraversalSignal::Skip
                    }
                }
            } else {
                TraversalSignal::Traverse
            }
        }
    }

    #[pyclass(module = "ruff_external", name = "Binding", unsendable)]
    #[allow(clippy::struct_excessive_bools)]
    pub(crate) struct PyBinding {
        #[pyo3(get)]
        id: u32,
        #[pyo3(get)]
        name: String,
        #[pyo3(get)]
        kind: String,
        #[pyo3(get)]
        range: (u32, u32),
        #[pyo3(get)]
        scope_id: u32,
        #[pyo3(get)]
        context: String,
        #[pyo3(get)]
        is_global: bool,
        #[pyo3(get)]
        is_nonlocal: bool,
        #[pyo3(get)]
        is_builtin: bool,
        #[pyo3(get)]
        is_used: bool,
        #[pyo3(get)]
        is_alias: bool,
        #[pyo3(get)]
        is_external: bool,
        #[pyo3(get)]
        is_explicit_export: bool,
        #[pyo3(get)]
        is_type_alias: bool,
        #[pyo3(get)]
        is_unbound: bool,
        #[pyo3(get)]
        is_unpacked_assignment: bool,
        #[pyo3(get)]
        is_private: bool,
        #[pyo3(get)]
        in_exception_handler: bool,
        #[pyo3(get)]
        in_assert_statement: bool,
    }

    impl PyBinding {
        fn new(checker: &Checker<'_>, binding_id: BindingId) -> Self {
            let semantic = checker.semantic();
            let binding = semantic.binding(binding_id);
            let name = binding.name(checker.source()).to_string();
            let range = (binding.range.start().to_u32(), binding.range.end().to_u32());
            PyBinding {
                id: binding_id.as_u32(),
                name,
                kind: binding_kind_name(&binding.kind).to_string(),
                range,
                scope_id: binding.scope.as_u32(),
                context: match binding.context {
                    ExecutionContext::Runtime => "runtime",
                    ExecutionContext::Typing => "typing",
                }
                .to_string(),
                is_global: binding.is_global(),
                is_nonlocal: binding.is_nonlocal(),
                is_builtin: matches!(binding.kind, BindingKind::Builtin),
                is_used: binding.is_used(),
                is_alias: binding.is_alias(),
                is_external: binding.is_external(),
                is_explicit_export: binding.is_explicit_export(),
                is_type_alias: binding.is_type_alias(),
                is_unbound: binding.is_unbound(),
                is_unpacked_assignment: binding.is_unpacked_assignment(),
                is_private: binding.is_private_declaration(),
                in_exception_handler: binding.in_exception_handler(),
                in_assert_statement: binding.in_assert_statement(),
            }
        }

        fn binding_id(&self) -> BindingId {
            BindingId::from_u32(self.id)
        }
    }

    #[pymethods]
    impl PyBinding {
        fn statement(
            &self,
            py: Python<'_>,
            semantic: &SemanticModelView,
        ) -> PyResult<Option<PyObject>> {
            semantic.with_checker(|checker| {
                let semantic_model = checker.semantic();
                let binding = semantic_model.binding(self.binding_id());
                let Some(stmt) = binding.statement(semantic_model) else {
                    return Ok(None);
                };
                stmt_to_python(py, checker.locator(), stmt, semantic.projection).map(Some)
            })
        }

        fn expression(
            &self,
            py: Python<'_>,
            semantic: &SemanticModelView,
        ) -> PyResult<Option<PyObject>> {
            semantic.with_checker(|checker| {
                let semantic_model = checker.semantic();
                let binding = semantic_model.binding(self.binding_id());
                let Some(expr) = binding.expression(semantic_model) else {
                    return Ok(None);
                };
                expr_to_python(py, checker.locator(), expr, semantic.projection).map(Some)
            })
        }

        fn references(&self, semantic: &SemanticModelView) -> PyResult<Vec<PyReference>> {
            semantic.with_checker(|checker| {
                let semantic_model = checker.semantic();
                let binding = semantic_model.binding(self.binding_id());
                Ok(binding
                    .references()
                    .map(|reference_id| {
                        let reference = semantic_model.reference(reference_id);
                        PyReference::new(reference_id, reference)
                    })
                    .collect())
            })
        }
    }

    #[pyclass(module = "ruff_external", name = "Reference", unsendable)]
    #[allow(clippy::struct_excessive_bools)]
    pub(crate) struct PyReference {
        #[pyo3(get)]
        id: u32,
        #[pyo3(get)]
        range: (u32, u32),
        #[pyo3(get)]
        node_id: Option<u32>,
        #[pyo3(get)]
        scope_id: u32,
        #[pyo3(get)]
        is_load: bool,
        #[pyo3(get)]
        in_typing_context: bool,
        #[pyo3(get)]
        in_runtime_context: bool,
        #[pyo3(get)]
        in_typing_only_annotation: bool,
        #[pyo3(get)]
        in_runtime_evaluated_annotation: bool,
        #[pyo3(get)]
        in_type_definition: bool,
        #[pyo3(get)]
        in_type_checking_block: bool,
        #[pyo3(get)]
        in_string_type_definition: bool,
        #[pyo3(get)]
        in_dunder_all_definition: bool,
        #[pyo3(get)]
        in_annotated_type_alias_value: bool,
        #[pyo3(get)]
        in_assert_statement: bool,
    }

    impl PyReference {
        fn new(reference_id: ResolvedReferenceId, reference: &ResolvedReference) -> Self {
            PyReference {
                id: reference_id.as_u32(),
                range: (reference.start().to_u32(), reference.end().to_u32()),
                node_id: reference.expression_id().map(NodeId::as_u32),
                scope_id: reference.scope_id().as_u32(),
                is_load: reference.is_load(),
                in_typing_context: reference.in_typing_context(),
                in_runtime_context: reference.in_runtime_context(),
                in_typing_only_annotation: reference.in_typing_only_annotation(),
                in_runtime_evaluated_annotation: reference.in_runtime_evaluated_annotation(),
                in_type_definition: reference.in_type_definition(),
                in_type_checking_block: reference.in_type_checking_block(),
                in_string_type_definition: reference.in_string_type_definition(),
                in_dunder_all_definition: reference.in_dunder_all_definition(),
                in_annotated_type_alias_value: reference.in_annotated_type_alias_value(),
                in_assert_statement: reference.in_assert_statement(),
            }
        }
    }

    #[pyclass(module = "ruff_external", name = "ScopeHandle", unsendable)]
    pub(crate) struct PyScopeHandle {
        #[pyo3(get)]
        id: u32,
        #[pyo3(get)]
        kind: String,
        #[pyo3(get)]
        parent_id: Option<u32>,
        #[pyo3(get)]
        node_id: Option<u32>,
        #[pyo3(get)]
        is_async: bool,
        #[pyo3(get)]
        uses_locals: bool,
    }

    impl PyScopeHandle {
        fn new(scope_id: ScopeId, scope: &Scope<'_>) -> Self {
            PyScopeHandle {
                id: scope_id.as_u32(),
                kind: scope_kind_name(&scope.kind).to_string(),
                parent_id: scope.parent.map(ScopeId::as_u32),
                node_id: scope_kind_node_id(&scope.kind),
                is_async: scope_kind_is_async(&scope.kind),
                uses_locals: scope.uses_locals(),
            }
        }
    }

    struct SemanticState {
        checker: Cell<Option<*const Checker<'static>>>,
    }

    impl SemanticState {
        fn new(checker: &Checker<'_>) -> Self {
            Self {
                checker: Cell::new(Some(ptr::from_ref(checker).cast::<Checker<'static>>())),
            }
        }

        fn deactivate(&self) {
            self.checker.set(None);
        }

        fn is_active(&self) -> bool {
            self.checker.get().is_some()
        }

        #[allow(unsafe_code)]
        fn with_checker<R>(&self, f: impl FnOnce(&Checker<'_>) -> PyResult<R>) -> PyResult<R> {
            let ptr = self.checker.get().ok_or_else(|| {
                PyRuntimeError::new_err(
                    "SemanticModel is no longer active; store results during the callback",
                )
            })?;
            let checker = unsafe { &*ptr.cast::<Checker<'_>>() };
            f(checker)
        }
    }

    fn resolve_name_string(checker: &Checker<'_>, node_id: NodeId) -> Option<String> {
        resolve_binding_id(checker, node_id).map(|binding_id| binding_name(checker, binding_id))
    }

    fn resolve_name_binding(checker: &Checker<'_>, node_id: NodeId) -> Option<PyBinding> {
        resolve_binding_id(checker, node_id).map(|binding_id| PyBinding::new(checker, binding_id))
    }

    fn resolve_binding_id(checker: &Checker<'_>, node_id: NodeId) -> Option<BindingId> {
        let semantic = checker.semantic();
        let expr = semantic.node(node_id).as_expression()?;
        let Expr::Name(name) = expr else {
            return None;
        };
        semantic
            .resolve_name(name)
            .or_else(|| binding_id_for_store(checker, name))
    }

    fn binding_id_for_store(
        checker: &Checker<'_>,
        name: &ruff_python_ast::ExprName,
    ) -> Option<BindingId> {
        if matches!(name.ctx, ExprContext::Store) {
            checker.semantic().current_scope().get(name.id.as_str())
        } else {
            None
        }
    }

    fn binding_name(checker: &Checker<'_>, binding_id: BindingId) -> String {
        checker
            .semantic()
            .binding(binding_id)
            .name(checker.source())
            .to_string()
    }

    fn binding_kind_name(kind: &BindingKind<'_>) -> &'static str {
        match kind {
            BindingKind::Annotation => "annotation",
            BindingKind::Argument => "argument",
            BindingKind::NamedExprAssignment => "named_expr_assignment",
            BindingKind::Assignment => "assignment",
            BindingKind::TypeParam => "type_param",
            BindingKind::LoopVar => "loop_var",
            BindingKind::WithItemVar => "with_item_var",
            BindingKind::Global(_) => "global",
            BindingKind::Nonlocal(..) => "nonlocal",
            BindingKind::Builtin => "builtin",
            BindingKind::ClassDefinition(_) => "class_definition",
            BindingKind::FunctionDefinition(_) => "function_definition",
            BindingKind::Export(_) => "export",
            BindingKind::FutureImport => "future_import",
            BindingKind::Import(_) => "import",
            BindingKind::FromImport(_) => "from_import",
            BindingKind::SubmoduleImport(_) => "submodule_import",
            BindingKind::Deletion => "deletion",
            BindingKind::ConditionalDeletion(_) => "conditional_deletion",
            BindingKind::BoundException => "bound_exception",
            BindingKind::UnboundException(_) => "unbound_exception",
            BindingKind::DunderClassCell => "__class__",
        }
    }

    fn scope_kind_name(kind: &ScopeKind<'_>) -> &'static str {
        match kind {
            ScopeKind::Module => "module",
            ScopeKind::Class(_) => "class",
            ScopeKind::Function(_) => "function",
            ScopeKind::Lambda(_) => "lambda",
            ScopeKind::Generator { kind, .. } => match kind {
                GeneratorKind::Generator => "generator",
                GeneratorKind::ListComprehension => "list_comprehension",
                GeneratorKind::DictComprehension => "dict_comprehension",
                GeneratorKind::SetComprehension => "set_comprehension",
            },
            ScopeKind::Type => "type",
            ScopeKind::DunderClassCell => "dunder_class_cell",
        }
    }

    fn scope_kind_is_async(kind: &ScopeKind<'_>) -> bool {
        match kind {
            ScopeKind::Function(stmt) => stmt.is_async,
            ScopeKind::Generator { is_async, .. } => *is_async,
            _ => false,
        }
    }

    fn scope_kind_node_id(kind: &ScopeKind<'_>) -> Option<u32> {
        match kind {
            ScopeKind::Class(stmt) => stmt.node_index().load().as_u32(),
            ScopeKind::Function(stmt) => stmt.node_index().load().as_u32(),
            ScopeKind::Lambda(expr) => expr.node_index().load().as_u32(),
            ScopeKind::Module
            | ScopeKind::Generator { .. }
            | ScopeKind::Type
            | ScopeKind::DunderClassCell => None,
        }
    }

    const MISSING_SEMANTIC_ID_MESSAGE: &str =
        "SemanticModel: node has no semantic id yet; use deferred targets (e.g., LambdaDeferred)";

    /// Return the semantic node ID for a given Python node, if available.
    ///
    /// Ruff does not assign semantic IDs to every AST node (for example, many nodes in typing-only
    /// contexts). For APIs that return `Optional[...]`, treat missing IDs as "unresolved" rather
    /// than raising.
    fn maybe_node_id(node: &Bound<'_, PyAny>, checker: &Checker<'_>) -> PyResult<Option<NodeId>> {
        let handle = node_handle(node)?;
        let semantic = checker.semantic();

        let Some(node_id) = handle.store.semantic_index(handle.node_id)? else {
            return Ok(None);
        };
        if !semantic.contains_node(node_id) {
            return Ok(None);
        }

        // Ensure the Python node refers to the same underlying AST node as the semantic model.
        let stored_index_ptr = handle.store.node_index_ptr(handle.node_id)?;
        let stored_index_ptr = stored_index_ptr.as_ptr();
        let semantic_index_ptr = match *semantic.node(node_id) {
            NodeRef::Stmt(stmt) => ptr::from_ref(stmt.node_index()),
            NodeRef::Expr(expr) => ptr::from_ref(expr.node_index()),
        };
        if ptr::eq(stored_index_ptr, semantic_index_ptr) {
            Ok(Some(node_id))
        } else {
            Err(PyValueError::new_err(MISSING_SEMANTIC_ID_MESSAGE))
        }
    }

    fn checked_node_id(node: &Bound<'_, PyAny>, checker: &Checker<'_>) -> PyResult<NodeId> {
        maybe_node_id(node, checker)?
            .ok_or_else(|| PyValueError::new_err(MISSING_SEMANTIC_ID_MESSAGE))
    }

    fn module_from_name(name: &str) -> Option<Modules> {
        match name {
            "_typeshed" => Some(Modules::TYPESHED),
            "anyio" => Some(Modules::ANYIO),
            "builtins" => Some(Modules::BUILTINS),
            "collections" => Some(Modules::COLLECTIONS),
            "copy" => Some(Modules::COPY),
            "contextvars" => Some(Modules::CONTEXTVARS),
            "dataclasses" => Some(Modules::DATACLASSES),
            "datetime" => Some(Modules::DATETIME),
            "django" => Some(Modules::DJANGO),
            "fastapi" => Some(Modules::FASTAPI),
            "flask" => Some(Modules::FLASK),
            "hashlib" => Some(Modules::HASHLIB),
            "logging" => Some(Modules::LOGGING),
            "markupsafe" => Some(Modules::MARKUPSAFE),
            "mock" => Some(Modules::MOCK),
            "numpy" => Some(Modules::NUMPY),
            "os" => Some(Modules::OS),
            "pandas" => Some(Modules::PANDAS),
            "pytest" => Some(Modules::PYTEST),
            "re" => Some(Modules::RE),
            "regex" => Some(Modules::REGEX),
            "six" => Some(Modules::SIX),
            "subprocess" => Some(Modules::SUBPROCESS),
            "tarfile" => Some(Modules::TARFILE),
            "trio" => Some(Modules::TRIO),
            "typing" => Some(Modules::TYPING),
            "typing_extensions" => Some(Modules::TYPING_EXTENSIONS),
            "attr" | "attrs" => Some(Modules::ATTRS),
            "airflow" => Some(Modules::AIRFLOW),
            "crypt" => Some(Modules::CRYPT),
            _ => None,
        }
    }
}
