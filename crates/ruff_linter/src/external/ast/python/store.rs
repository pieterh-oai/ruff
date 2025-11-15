use std::any::TypeId;
use std::cell::{Cell, RefCell};
use std::ptr::NonNull;
use std::rc::Rc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use ruff_python_ast::{AtomicNodeIndex, HasNodeIndex};
use ruff_python_semantic::NodeId;
use rustc_hash::FxHashMap;

thread_local! {
    static CURRENT_STORE: RefCell<Option<AstStoreHandle>> = const { RefCell::new(None) };
}

/// In-memory cache of AST nodes keyed by their node IDs.
#[derive(Debug, Default)]
struct AstStore {
    nodes: RefCell<FxHashMap<u32, StoredNode>>,
    ids_by_index: RefCell<FxHashMap<NonNull<AtomicNodeIndex>, u32>>,
    next_id: Cell<u32>,
    active: Cell<bool>,
}

#[derive(Clone, Copy, Debug)]
struct StoredNode {
    ptr: NonNull<()>,
    type_id: TypeId,
    index: NonNull<AtomicNodeIndex>,
}

#[derive(Clone, Debug)]
pub(crate) struct AstStoreHandle {
    inner: Rc<AstStore>,
}

impl AstStoreHandle {
    pub(crate) fn new() -> Self {
        let handle = Self {
            inner: Rc::new(AstStore::default()),
        };
        handle.inner.active.set(true);
        handle
    }

    /// Returns the node ID, assigning one if it hasn't been set yet, and ensures the node is
    /// stored for later lookup.
    pub(crate) fn ensure<T>(&self, node: &T) -> u32
    where
        T: HasNodeIndex + 'static,
    {
        let id = self.inner.ensure_id_for_index(node.node_index());
        let mut nodes = self.inner.nodes.borrow_mut();
        let ptr = NonNull::from(node).cast::<()>();
        let type_id = TypeId::of::<T>();
        let index = NonNull::from(node.node_index());
        nodes.entry(id).or_insert(StoredNode {
            ptr,
            type_id,
            index,
        });
        id
    }

    pub(crate) fn ensure_id_for_semantic_index(&self, node_index: &AtomicNodeIndex) -> u32 {
        self.inner.ensure_id_for_index(node_index)
    }

    #[allow(
        unsafe_code,
        reason = "This store is invalidated before the underlying AST nodes are dropped."
    )]
    pub(crate) fn get<T>(&self, id: u32) -> PyResult<&T>
    where
        T: 'static,
    {
        if !self.inner.active.get() {
            return Err(PyRuntimeError::new_err("AST store is no longer valid"));
        }
        let nodes = self.inner.nodes.borrow();
        let Some(&StoredNode { ptr, type_id, .. }) = nodes.get(&id) else {
            return Err(PyRuntimeError::new_err(format!(
                "missing AST node for id {id}"
            )));
        };
        if type_id != TypeId::of::<T>() {
            return Err(PyRuntimeError::new_err(format!(
                "type mismatch for AST node {id}"
            )));
        }
        // SAFETY: Entries are only inserted via `ensure`, which stores a pointer to a `T`.
        // We rely on `invalidate` being called before the underlying AST data can be dropped.
        let reference = unsafe { ptr.cast::<T>().as_ref() };
        Ok(reference)
    }

    #[allow(
        unsafe_code,
        reason = "This store is invalidated before the underlying AST nodes are dropped."
    )]
    pub(crate) fn semantic_index(&self, id: u32) -> PyResult<NodeId> {
        if !self.inner.active.get() {
            return Err(PyRuntimeError::new_err("AST store is no longer valid"));
        }
        let nodes = self.inner.nodes.borrow();
        let Some(&StoredNode { index, .. }) = nodes.get(&id) else {
            return Err(PyRuntimeError::new_err(format!(
                "missing AST node for id {id}"
            )));
        };
        let node_index = unsafe { index.as_ref().load() };
        let Some(node_id) = node_index.as_u32() else {
            return Err(PyRuntimeError::new_err(format!(
                "node id missing for AST node {id}"
            )));
        };
        Ok(NodeId::from_u32(node_id))
    }

    pub(crate) fn invalidate(&self) {
        self.inner.invalidate();
    }
}

impl AstStore {
    fn ensure_id_for_index(&self, index: &AtomicNodeIndex) -> u32 {
        debug_assert!(self.active.get(), "ensuring id on inactive store");
        let index_ptr = NonNull::from(index);

        if let Some(&id) = self.ids_by_index.borrow().get(&index_ptr) {
            return id;
        }

        let id = self.next_id.get();
        self.next_id
            .set(id.checked_add(1).expect("exceeded maximum node id"));
        self.ids_by_index.borrow_mut().insert(index_ptr, id);
        id
    }

    fn invalidate(&self) {
        self.nodes.borrow_mut().clear();
        self.ids_by_index.borrow_mut().clear();
        self.active.set(false);
    }
}

pub(crate) fn with_store<R>(store: AstStoreHandle, f: impl FnOnce() -> R) -> R {
    struct Restore<'a>(&'a RefCell<Option<AstStoreHandle>>, Option<AstStoreHandle>);

    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            self.0.replace(self.1.take());
        }
    }

    CURRENT_STORE.with(|cell| {
        let previous = cell.replace(Some(store));
        let _restore = Restore(cell, previous);
        f()
    })
}

pub(crate) fn current_store() -> AstStoreHandle {
    CURRENT_STORE.with(|cell| {
        cell.borrow()
            .clone()
            .expect("AstStoreHandle requested without a current store")
    })
}
