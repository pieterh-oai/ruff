# NOTE: Keep this file in sync with `crates/ruff_linter/src/external/ast/target.rs`.
from __future__ import annotations

from typing import Any, Callable, TypedDict

from .nodes import (
    AnnAssignStmt,
    AssertStmt,
    AssignStmt,
    AttributeExpr,
    AugAssignStmt,
    AwaitExpr,
    BinOpExpr,
    BoolOpExpr,
    BooleanLiteralExpr,
    BreakStmt,
    BytesLiteralExpr,
    CallExpr,
    ClassDefStmt,
    CompareExpr,
    ContinueStmt,
    DeleteStmt,
    DictCompExpr,
    DictExpr,
    EllipsisLiteralExpr,
    ExprStmt,
    FStringExpr,
    ForStmt,
    FunctionDefStmt,
    GeneratorExpr,
    GlobalStmt,
    IfExpr,
    IfStmt,
    ImportFromStmt,
    ImportStmt,
    IpyEscapeCommandExpr,
    IpyEscapeCommandStmt,
    LambdaExpr,
    ListCompExpr,
    ListExpr,
    MatchStmt,
    NameExpr,
    NamedExpr,
    NoneLiteralExpr,
    NonlocalStmt,
    NumberLiteralExpr,
    PassStmt,
    RaiseStmt,
    ReturnStmt,
    SetCompExpr,
    SetExpr,
    SliceExpr,
    StarredExpr,
    StringLiteralExpr,
    SubscriptExpr,
    TStringExpr,
    TryStmt,
    TupleExpr,
    TypeAliasStmt,
    UnaryOpExpr,
    WhileStmt,
    WithStmt,
    YieldExpr,
    YieldFromExpr,
)

__all__ = ['VisitHandlers']

class VisitHandlers(TypedDict, total=False):
    stmt_FunctionDef: Callable[[FunctionDefStmt], Any]
    stmt_FunctionDefDeferred: Callable[[FunctionDefStmt], Any]
    stmt_ClassDef: Callable[[ClassDefStmt], Any]
    stmt_Return: Callable[[ReturnStmt], Any]
    stmt_Delete: Callable[[DeleteStmt], Any]
    stmt_TypeAlias: Callable[[TypeAliasStmt], Any]
    stmt_Assign: Callable[[AssignStmt], Any]
    stmt_AugAssign: Callable[[AugAssignStmt], Any]
    stmt_AnnAssign: Callable[[AnnAssignStmt], Any]
    stmt_For: Callable[[ForStmt], Any]
    stmt_While: Callable[[WhileStmt], Any]
    stmt_If: Callable[[IfStmt], Any]
    stmt_With: Callable[[WithStmt], Any]
    stmt_Match: Callable[[MatchStmt], Any]
    stmt_Raise: Callable[[RaiseStmt], Any]
    stmt_Try: Callable[[TryStmt], Any]
    stmt_Assert: Callable[[AssertStmt], Any]
    stmt_Import: Callable[[ImportStmt], Any]
    stmt_ImportFrom: Callable[[ImportFromStmt], Any]
    stmt_Global: Callable[[GlobalStmt], Any]
    stmt_Nonlocal: Callable[[NonlocalStmt], Any]
    stmt_Expr: Callable[[ExprStmt], Any]
    stmt_Pass: Callable[[PassStmt], Any]
    stmt_Break: Callable[[BreakStmt], Any]
    stmt_Continue: Callable[[ContinueStmt], Any]
    stmt_IpyEscapeCommand: Callable[[IpyEscapeCommandStmt], Any]
    expr_Attribute: Callable[[AttributeExpr], Any]
    expr_Await: Callable[[AwaitExpr], Any]
    expr_BinOp: Callable[[BinOpExpr], Any]
    expr_BoolOp: Callable[[BoolOpExpr], Any]
    expr_BooleanLiteral: Callable[[BooleanLiteralExpr], Any]
    expr_BytesLiteral: Callable[[BytesLiteralExpr], Any]
    expr_Call: Callable[[CallExpr], Any]
    expr_Compare: Callable[[CompareExpr], Any]
    expr_Dict: Callable[[DictExpr], Any]
    expr_DictComp: Callable[[DictCompExpr], Any]
    expr_EllipsisLiteral: Callable[[EllipsisLiteralExpr], Any]
    expr_FString: Callable[[FStringExpr], Any]
    expr_Generator: Callable[[GeneratorExpr], Any]
    expr_If: Callable[[IfExpr], Any]
    expr_IpyEscapeCommand: Callable[[IpyEscapeCommandExpr], Any]
    expr_Lambda: Callable[[LambdaExpr], Any]
    expr_LambdaDeferred: Callable[[LambdaExpr], Any]
    expr_List: Callable[[ListExpr], Any]
    expr_ListComp: Callable[[ListCompExpr], Any]
    expr_Name: Callable[[NameExpr], Any]
    expr_Named: Callable[[NamedExpr], Any]
    expr_NoneLiteral: Callable[[NoneLiteralExpr], Any]
    expr_NumberLiteral: Callable[[NumberLiteralExpr], Any]
    expr_Set: Callable[[SetExpr], Any]
    expr_SetComp: Callable[[SetCompExpr], Any]
    expr_Slice: Callable[[SliceExpr], Any]
    expr_Starred: Callable[[StarredExpr], Any]
    expr_StringLiteral: Callable[[StringLiteralExpr], Any]
    expr_Subscript: Callable[[SubscriptExpr], Any]
    expr_Tuple: Callable[[TupleExpr], Any]
    expr_TString: Callable[[TStringExpr], Any]
    expr_UnaryOp: Callable[[UnaryOpExpr], Any]
    expr_Yield: Callable[[YieldExpr], Any]
    expr_YieldFrom: Callable[[YieldFromExpr], Any]
