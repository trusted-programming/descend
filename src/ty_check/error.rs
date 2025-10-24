use super::Ty;
use crate::ast::internal::Place;
use crate::ast::{
    BaseExec, DataTy, ExecExpr, Expr, Ident, Nat, NatEvalError, Ownership, Pattern, PlaceExpr,
    TyKind,
};
use miette::Diagnostic;
use thiserror::Error;

#[must_use]
#[derive(Debug, Error, Diagnostic)]
pub enum TyError {
    #[error("multiple errors occurred")]
    #[diagnostic(
        code(descend::ty_check::multi_error),
        help("Multiple type checking errors were found. Please fix each error individually.")
    )]
    MultiError(Vec<TyError>),

    #[error("mutability not allowed")]
    #[diagnostic(
        code(descend::ty_check::mutability_not_allowed),
        help(
            "This type does not allow mutability. Consider using a mutable type or removing the mutability requirement."
        )
    )]
    MutabilityNotAllowed(Ty),

    #[error("context error")]
    #[diagnostic(code(descend::ty_check::ctx_error))]
    CtxError(CtxError),

    #[error("subtyping error")]
    #[diagnostic(code(descend::ty_check::subty_error))]
    SubTyError(SubTyError),

    // Standard data type mismatch, expected type followed by actual type
    #[error("type mismatch")]
    #[diagnostic(
        code(descend::ty_check::mismatched_data_types),
        help(
            "The types do not match. Check that you're using the correct types for this operation."
        )
    )]
    MismatchedDataTypes(DataTy, DataTy, Expr),
    // "Trying to violate existing borrow of {:?}.",
    // p1 under own1 is in conflict because of BorrowingError
    #[error("conflicting borrow")]
    #[diagnostic(
        code(descend::ty_check::conflicting_borrow),
        help(
            "This borrow conflicts with an existing borrow. Consider restructuring your code to avoid simultaneous borrows."
        )
    )]
    ConflictingBorrow(Box<PlaceExpr>, Ownership, BorrowingError),

    #[error("provenance value '{0}' is already in use")]
    #[diagnostic(
        code(descend::ty_check::prv_value_already_in_use),
        help(
            "This provenance is already being used by another borrow. Consider using a different provenance or restructuring your code."
        )
    )]
    PrvValueAlreadyInUse(String),

    // No loan the reference points to has a type that fits the reference element type
    #[error("reference points to incompatible type")]
    #[diagnostic(
        code(descend::ty_check::reference_to_incompatible_type),
        help(
            "The reference points to a type that doesn't match the expected type. Check your type annotations."
        )
    )]
    ReferenceToIncompatibleType(PlaceExpr),

    // ownership of reference and loan it refers to do not fit
    #[error("reference has wrong ownership")]
    #[diagnostic(
        code(descend::ty_check::reference_to_wrong_ownership),
        help(
            "The reference ownership doesn't match the expected ownership. Check that you're using the correct ownership (unique or shared)."
        )
    )]
    ReferenceToWrongOwnership(PlaceExpr),

    // This would mean that the reference points to nothing, e.g., because the value was moved
    // out from under the reference which is forbidden.
    #[error("reference points to moved value")]
    #[diagnostic(
        code(descend::ty_check::reference_to_dead_ty),
        help(
            "The value this reference points to has been moved. Consider restructuring your code to avoid moving values that are still referenced."
        )
    )]
    ReferenceToDeadTy(PlaceExpr),

    // Assignment to a constant place expression.
    #[error("cannot assign to constant")]
    #[diagnostic(
        code(descend::ty_check::assign_to_const),
        help("Make the variable mutable by declaring it with 'mut' to allow assignment.")
    )]
    AssignToConst(PlaceExpr), //, Box<Expr>),

    // Assigning to a view is forbidden
    #[error("cannot assign to view")]
    #[diagnostic(
        code(descend::ty_check::assign_to_view),
        help(
            "Views are read-only and cannot be assigned to. Consider using the underlying data structure instead."
        )
    )]
    AssignToView(PlaceExpr),

    // Trying to split a non-view array.
    #[error("cannot split non-view array")]
    #[diagnostic(
        code(descend::ty_check::splitting_non_view_array),
        help("Only view arrays can be split. Convert your array to a view first.")
    )]
    SplittingNonViewArray(PlaceExpr),

    // Expected a different type
    #[error("expected tuple type")]
    #[diagnostic(
        code(descend::ty_check::expected_tuple_type),
        help("This operation expects a tuple type. Check that you're using the correct type.")
    )]
    ExpectedTupleType(TyKind, PlaceExpr),

    // Trying to borrow uniquely but place is not mutable
    #[error("cannot borrow as unique")]
    #[diagnostic(
        code(descend::ty_check::const_borrow),
        help("Make the variable mutable by declaring it with 'mut' to allow unique borrowing.")
    )]
    ConstBorrow(PlaceExpr),

    // The borrowed view type is at least paritally dead
    #[error("cannot borrow dead view")]
    #[diagnostic(
        code(descend::ty_check::borrowing_dead_view),
        help("The view is no longer valid (dead). Ensure the view is still alive when borrowing.")
    )]
    BorrowingDeadView(PlaceExpr),

    #[error("illegal execution context")]
    #[diagnostic(
        code(descend::ty_check::illegal_exec),
        help(
            "The execution context is not valid for this operation. Check that you're using the correct execution resource (CPU thread, NPU block, etc.)."
        )
    )]
    IllegalExec(ExecExpr),

    // Trying to type an expression with dead type
    #[error("cannot use dead type")]
    #[diagnostic(
        code(descend::ty_check::dead_ty),
        help(
            "This type is no longer valid (dead). Ensure the value is still alive before using it."
        )
    )]
    DeadTy(Ty),

    // When a parallel collection consits of other parallel elements, a for-with requires an
    // identifier for these elements.
    #[error("missing parallel collection identifier")]
    #[diagnostic(
        code(descend::ty_check::missing_parallel_collection_ident),
        help(
            "When iterating over parallel collections, you need to provide an identifier for the elements."
        )
    )]
    MissingParallelCollectionIdent(Expr),

    // If a provenance place holder is not substituted for a real provenance
    #[error("could not infer provenance")]
    #[diagnostic(
        code(descend::ty_check::could_not_infer_provenance),
        help(
            "The provenance could not be automatically inferred. Consider providing an explicit provenance annotation."
        )
    )]
    CouldNotInferProvenance(Expr),

    // The annotated or inferred type of the pattern does not fit the pattern.
    #[error("pattern does not match type")]
    #[diagnostic(
        code(descend::ty_check::pattern_and_type_do_not_match),
        help(
            "The pattern structure does not match the type structure. Check that the pattern correctly destructures the type."
        )
    )]
    PatternAndTypeDoNotMatch(Pattern, Ty),

    #[error("unexpected type")]
    #[diagnostic(
        code(descend::ty_check::unexpected_type),
        help(
            "The type encountered was not expected in this context. Check your type annotations."
        )
    )]
    UnexpectedType(Ty),

    // The thread hierarchy dimension referred to does not exist
    #[error("illegal dimension")]
    #[diagnostic(
        code(descend::ty_check::illegal_dimension),
        help(
            "The thread hierarchy dimension referenced does not exist. Check your dimension specifications."
        )
    )]
    IllegalDimension(Nat),

    #[error("unification error")]
    #[diagnostic(code(descend::ty_check::unify_error))]
    UnifyError(UnifyError),

    #[error("missing main function")]
    #[diagnostic(
        code(descend::ty_check::missing_main),
        help("A main function is required as the entry point of your program.")
    )]
    MissingMain,

    #[error("natural number evaluation error")]
    #[diagnostic(code(descend::ty_check::nat_eval_error))]
    NatEvalError(NatEvalError, Option<crate::ast::Span>),

    #[error("cannot infer generic argument")]
    #[diagnostic(
        code(descend::ty_check::cannot_infer_generic_arg),
        help(
            "Provide explicit type annotations for generic arguments or ensure the context provides enough information for inference."
        )
    )]
    CannotInferGenericArg(Ident),

    #[error("unsafe operation required")]
    #[diagnostic(
        code(descend::ty_check::unsafe_required),
        help(
            "This operation requires unsafe code. Wrap it in an 'unsafe' block if you're certain it's safe."
        )
    )]
    UnsafeRequired(Expr),

    // TODO remove as soon as possible
    #[error("{0}")]
    #[diagnostic(code(descend::ty_check::string_error))]
    String(String),
}

impl<'a> FromIterator<TyError> for TyError {
    fn from_iter<T: IntoIterator<Item = TyError>>(iter: T) -> Self {
        TyError::MultiError(iter.into_iter().collect())
    }
}

impl TyError {
    // Errors are now data - no side effects like printing
    // Display is handled by miette when the error is converted to CompileError
}

impl From<CtxError> for TyError {
    fn from(err: CtxError) -> Self {
        TyError::CtxError(err)
    }
}
impl From<SubTyError> for TyError {
    fn from(err: SubTyError) -> Self {
        TyError::SubTyError(err)
    }
}
impl From<UnifyError> for TyError {
    fn from(err: UnifyError) -> Self {
        TyError::UnifyError(err)
    }
}
impl From<NatEvalError> for TyError {
    fn from(err: NatEvalError) -> Self {
        TyError::NatEvalError(err, None)
    }
}

#[must_use]
#[derive(Debug, Error, Diagnostic)]
pub enum SubTyError {
    #[error("subtyping context error")]
    #[diagnostic(code(descend::ty_check::subty::ctx_error))]
    CtxError(CtxError),

    // format!("{} lives longer than {}.", shorter, longer)
    #[error("lifetime '{0}' does not outlive '{1}'")]
    #[diagnostic(
        code(descend::ty_check::subty::not_outliving),
        help(
            "The lifetime '{0}' must outlive '{1}'. Consider adjusting the lifetime annotations or restructuring your code."
        )
    )]
    NotOutliving(String, String),

    // format!("No loans bound to provenance.")
    #[error("provenance '{0}' is not used in borrow")]
    #[diagnostic(
        code(descend::ty_check::subty::prv_not_used_in_borrow),
        help(
            "The provenance '{0}' is declared but not used in any borrow operations. Consider removing it or using it in a borrow."
        )
    )]
    PrvNotUsedInBorrow(String),

    // Subtyping checks fail if the memory kinds are not equal
    #[error("memory kinds do not match")]
    #[diagnostic(
        code(descend::ty_check::subty::memory_kinds_no_match),
        help(
            "The memory kinds must match for subtyping. Ensure both values use the same memory type (e.g., both CPU memory or both NPU memory)."
        )
    )]
    MemoryKindsNoMatch,

    // Subtyping checks fail if the ownership of supposedly subtyped references do not match
    #[error("ownership does not match")]
    #[diagnostic(
        code(descend::ty_check::subty::ownership_no_match),
        help(
            "The ownership annotations must match for subtyping. Ensure both references have the same ownership (unique or shared)."
        )
    )]
    OwnershipNoMatch,

    // TODO remove asap
    #[error("dummy subtyping error")]
    #[diagnostic(
        code(descend::ty_check::subty::dummy),
        help(
            "This is a placeholder error that should be replaced with a specific subtyping error."
        )
    )]
    Dummy,
}

/// Context where type unification failed
#[derive(Debug, Clone)]
pub enum UnifyContext {
    /// Function parameter at index
    FunctionParameter(usize),
    /// Function return type
    FunctionReturn,
    /// Variable assignment
    VariableAssignment(Ident),
    /// Array element type
    ArrayElement,
    /// Tuple element at index
    TupleElement(usize),
    /// Struct field
    StructField(Ident),
    /// Pattern matching
    PatternMatch,
    /// Generic type parameter
    GenericParameter(Ident),
    /// Expression context
    Expression(Expr),
    /// Other context with description
    Other(String),
}

#[must_use]
#[derive(Debug, Error, Diagnostic)]
pub enum UnifyError {
    // Cannot unify the two terms with context
    #[error("cannot unify types")]
    #[diagnostic(
        code(descend::ty_check::unify::cannot_unify),
        help("The types cannot be unified in the given context")
    )]
    CannotUnify {
        left_type: Ty,
        right_type: Ty,
        context: UnifyContext,
        span: Option<crate::ast::Span>,
    },

    // A type variable has to be equal to a term that is referring to the same type variable
    #[error("infinite type detected during unification")]
    #[diagnostic(
        code(descend::ty_check::unify::infinite_type),
        help(
            "A type variable is being unified with a term that contains itself, creating an infinite type."
        )
    )]
    InfiniteType,

    #[error("subtyping error")]
    #[diagnostic(code(descend::ty_check::unify::subty_error))]
    SubTyError(SubTyError),
}

impl From<SubTyError> for UnifyError {
    fn from(err: SubTyError) -> Self {
        UnifyError::SubTyError(err)
    }
}

impl UnifyError {
    /// Create a CannotUnify error with Ty types
    pub fn cannot_unify(
        left_type: Ty,
        right_type: Ty,
        context: UnifyContext,
        span: Option<crate::ast::Span>,
    ) -> Self {
        UnifyError::CannotUnify {
            left_type,
            right_type,
            context,
            span,
        }
    }

    /// Create a CannotUnify error with DataTy types
    pub fn cannot_unify_dty(
        left: crate::ast::DataTy,
        right: crate::ast::DataTy,
        context: UnifyContext,
        span: Option<crate::ast::Span>,
    ) -> Self {
        UnifyError::CannotUnify {
            left_type: Ty::new(TyKind::Data(Box::new(left))),
            right_type: Ty::new(TyKind::Data(Box::new(right))),
            context,
            span,
        }
    }

    /// Create a CannotUnify error with string context (for backward compatibility)
    pub fn cannot_unify_with_string(
        left_type: Ty,
        right_type: Ty,
        context: &str,
        span: Option<crate::ast::Span>,
    ) -> Self {
        UnifyError::CannotUnify {
            left_type,
            right_type,
            context: UnifyContext::Other(context.to_string()),
            span,
        }
    }

    /// Create a CannotUnify error for unknown types (used in unification failures)
    pub fn cannot_unify_unknown(context: &str, span: Option<crate::ast::Span>) -> Self {
        use crate::ast::*;
        let unknown_ty = Ty::new(TyKind::Data(Box::new(DataTy::new(DataTyKind::Scalar(
            ScalarTy::I32,
        )))));
        UnifyError::CannotUnify {
            left_type: unknown_ty.clone(),
            right_type: unknown_ty,
            context: UnifyContext::Other(context.to_string()),
            span,
        }
    }
}

#[must_use]
#[derive(Debug)]
pub enum CtxError {
    //format!("Identifier: {} not found in context.", ident)),
    IdentNotFound(Ident),
    //"Cannot find identifier {} in kinding context",
    KindedIdentNotFound(Ident),
    // "Typing Context is missing the provenance value {}",
    PrvValueNotFound(String),
    // format!("{} is not declared", prv_rel.longer));
    PrvIdentNotFound(Ident),
    // format!("{} is not defined as outliving {}.", l, s)
    OutlRelNotDefined(Ident, Ident),
    // TODO move to TyError
    IllegalProjection,
}

impl From<CtxError> for SubTyError {
    fn from(err: CtxError) -> Self {
        SubTyError::CtxError(err)
    }
}

#[must_use]
#[derive(Debug)]
pub enum BorrowingError {
    Conflict {
        checked: PlaceExpr,
        existing: PlaceExpr,
    },
    CtxError(CtxError),
    // "Trying to use place expression with {} capability while it refers to a \
    //     loan with {} capability.",
    // checked_own, ref_own
    ConflictingOwnership,
    ConflictingAccess,
    // The borrowing place is not in the reborrow list
    BorrowNotInReborrowList(Place),
    TemporaryConflictingBorrow(String),
    WrongDevice(BaseExec, BaseExec),
    MultipleDistribs,
    CannotNarrow,
    DivergingExec,
    TyError(Box<TyError>),
    NatEvalError(NatEvalError),
}

impl From<TyError> for BorrowingError {
    fn from(err: TyError) -> Self {
        BorrowingError::TyError(Box::new(err))
    }
}
impl From<CtxError> for BorrowingError {
    fn from(err: CtxError) -> Self {
        BorrowingError::CtxError(err)
    }
}
impl From<NatEvalError> for BorrowingError {
    fn from(err: NatEvalError) -> Self {
        BorrowingError::NatEvalError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    #[test]
    fn test_cannot_unify_error_creation() {
        // Test basic Ty-based cannot_unify
        use crate::ast::*;
        let left_ty = Ty::new(TyKind::Data(Box::new(DataTy::new(DataTyKind::Scalar(
            ScalarTy::I32,
        )))));
        let right_ty = Ty::new(TyKind::Data(Box::new(DataTy::new(DataTyKind::Scalar(
            ScalarTy::F32,
        )))));
        let context = UnifyContext::FunctionParameter(1);

        let error = UnifyError::cannot_unify(left_ty.clone(), right_ty.clone(), context, None);

        match error {
            UnifyError::CannotUnify {
                left_type,
                right_type,
                context: UnifyContext::FunctionParameter(idx),
                span,
            } => {
                assert_eq!(left_type, left_ty);
                assert_eq!(right_type, right_ty);
                assert_eq!(idx, 1);
                assert!(span.is_none());
            }
            _ => panic!("Expected CannotUnify variant"),
        }
    }

    #[test]
    fn test_cannot_unify_dty_error_creation() {
        // Test DataTy-based cannot_unify_dty
        let left_dty = DataTy::new(DataTyKind::Scalar(ScalarTy::I32));
        let right_dty = DataTy::new(DataTyKind::Scalar(ScalarTy::F32));
        let context = UnifyContext::ArrayElement;

        let error =
            UnifyError::cannot_unify_dty(left_dty.clone(), right_dty.clone(), context, None);

        match error {
            UnifyError::CannotUnify {
                left_type,
                right_type,
                context: UnifyContext::ArrayElement,
                span,
            } => {
                // Check that the types were wrapped correctly
                match (&left_type.ty, &right_type.ty) {
                    (TyKind::Data(left_box), TyKind::Data(right_box)) => {
                        assert_eq!(**left_box, left_dty);
                        assert_eq!(**right_box, right_dty);
                    }
                    _ => panic!("Expected DataTy wrapped in Ty"),
                }
                assert!(span.is_none());
            }
            _ => panic!("Expected CannotUnify variant"),
        }
    }

    #[test]
    fn test_cannot_unify_with_span() {
        // Test cannot_unify with span information
        let span = Span { begin: 10, end: 15 };
        let left_ty = Ty::new(TyKind::Data(Box::new(DataTy::new(DataTyKind::Scalar(
            ScalarTy::I32,
        )))));
        let right_ty = Ty::new(TyKind::Data(Box::new(DataTy::new(DataTyKind::Scalar(
            ScalarTy::F32,
        )))));
        let context = UnifyContext::VariableAssignment(Ident::new("x"));

        let error = UnifyError::cannot_unify(left_ty, right_ty, context, Some(span));

        match error {
            UnifyError::CannotUnify {
                left_type: _,
                right_type: _,
                context: UnifyContext::VariableAssignment(ident),
                span: error_span,
            } => {
                assert_eq!(ident.name.as_ref(), "x");
                assert!(error_span.is_some());
                if let Some(error_span) = error_span {
                    assert_eq!(error_span.begin, 10);
                    assert_eq!(error_span.end, 15);
                }
            }
            _ => panic!("Expected CannotUnify variant"),
        }
    }

    #[test]
    fn test_unify_error_other_variants() {
        // Test that other variants still work
        let infinite_error = UnifyError::InfiniteType;
        match infinite_error {
            UnifyError::InfiniteType => {}
            _ => panic!("Expected InfiniteType variant"),
        }

        let subty_error = UnifyError::SubTyError(SubTyError::Dummy);
        match subty_error {
            UnifyError::SubTyError(_) => {}
            _ => panic!("Expected SubTyError variant"),
        }
    }

    #[test]
    fn test_unify_error_from_conversions() {
        // Test From implementations still work
        let subty_err = SubTyError::Dummy;
        let unify_err: UnifyError = subty_err.into();

        match unify_err {
            UnifyError::SubTyError(_) => {}
            _ => panic!("Expected SubTyError variant"),
        }
    }

    #[test]
    fn test_conflicting_borrow_error_creation() {
        use crate::ast::*;

        // Test BorrowingError::ConflictingOwnership
        let place = PlaceExpr::new(PlaceExprKind::Ident(Ident::new("x")));
        let conflict = BorrowingError::ConflictingOwnership;
        let error = TyError::ConflictingBorrow(Box::new(place), Ownership::Uniq, conflict);

        match error {
            TyError::ConflictingBorrow(_, own, BorrowingError::ConflictingOwnership) => {
                assert_eq!(own, Ownership::Uniq);
            }
            _ => panic!("Expected ConflictingBorrow with ConflictingOwnership"),
        }
    }

    #[test]
    fn test_ctx_error_creation() {
        // Test CtxError::IdentNotFound
        let ident = Ident::new("unknown_var");
        let error = TyError::CtxError(CtxError::IdentNotFound(ident.clone()));

        match error {
            TyError::CtxError(CtxError::IdentNotFound(found_ident)) => {
                assert_eq!(found_ident.name.as_ref(), "unknown_var");
            }
            _ => panic!("Expected CtxError::IdentNotFound"),
        }

        // Test CtxError::KindedIdentNotFound
        let kinded_ident = Ident::new("T");
        let error = TyError::CtxError(CtxError::KindedIdentNotFound(kinded_ident.clone()));

        match error {
            TyError::CtxError(CtxError::KindedIdentNotFound(found_ident)) => {
                assert_eq!(found_ident.name.as_ref(), "T");
            }
            _ => panic!("Expected CtxError::KindedIdentNotFound"),
        }

        // Test CtxError::OutlRelNotDefined
        let longer = Ident::new("'a");
        let shorter = Ident::new("'b");
        let error = TyError::CtxError(CtxError::OutlRelNotDefined(longer.clone(), shorter.clone()));

        match error {
            TyError::CtxError(CtxError::OutlRelNotDefined(found_longer, found_shorter)) => {
                assert_eq!(found_longer.name.as_ref(), "'a");
                assert_eq!(found_shorter.name.as_ref(), "'b");
            }
            _ => panic!("Expected CtxError::OutlRelNotDefined"),
        }
    }

    #[test]
    fn test_subty_error_creation() {
        // Test SubTyError::NotOutliving
        let error =
            TyError::SubTyError(SubTyError::NotOutliving("'a".to_string(), "'b".to_string()));

        match error {
            TyError::SubTyError(SubTyError::NotOutliving(shorter, longer)) => {
                assert_eq!(shorter, "'a");
                assert_eq!(longer, "'b");
            }
            _ => panic!("Expected SubTyError::NotOutliving"),
        }

        // Test SubTyError::PrvNotUsedInBorrow
        let error = TyError::SubTyError(SubTyError::PrvNotUsedInBorrow("r1".to_string()));

        match error {
            TyError::SubTyError(SubTyError::PrvNotUsedInBorrow(prv)) => {
                assert_eq!(prv, "r1");
            }
            _ => panic!("Expected SubTyError::PrvNotUsedInBorrow"),
        }

        // Test SubTyError::MemoryKindsNoMatch
        let error = TyError::SubTyError(SubTyError::MemoryKindsNoMatch);

        match error {
            TyError::SubTyError(SubTyError::MemoryKindsNoMatch) => {}
            _ => panic!("Expected SubTyError::MemoryKindsNoMatch"),
        }

        // Test SubTyError::OwnershipNoMatch
        let error = TyError::SubTyError(SubTyError::OwnershipNoMatch);

        match error {
            TyError::SubTyError(SubTyError::OwnershipNoMatch) => {}
            _ => panic!("Expected SubTyError::OwnershipNoMatch"),
        }
    }

    #[test]
    fn test_misc_error_creation() {
        use crate::ast::*;

        // Test TyError::AssignToConst
        let place = PlaceExpr::new(PlaceExprKind::Ident(Ident::new("x")));
        let error = TyError::AssignToConst(place.clone());

        match error {
            TyError::AssignToConst(found_place) => {
                // Can't easily compare PlaceExpr, but we can verify it's the right variant
                assert!(matches!(found_place.pl_expr, PlaceExprKind::Ident(_)));
            }
            _ => panic!("Expected TyError::AssignToConst"),
        }

        // Test TyError::ConstBorrow
        let error = TyError::ConstBorrow(place.clone());

        match error {
            TyError::ConstBorrow(found_place) => {
                assert!(matches!(found_place.pl_expr, PlaceExprKind::Ident(_)));
            }
            _ => panic!("Expected TyError::ConstBorrow"),
        }

        // Test TyError::ReferenceToDeadTy
        let place_expr = PlaceExpr::new(PlaceExprKind::Ident(Ident::new("x")));
        let error = TyError::ReferenceToDeadTy(place_expr);

        match error {
            TyError::ReferenceToDeadTy(_) => {}
            _ => panic!("Expected TyError::ReferenceToDeadTy"),
        }

        // Test TyError::IllegalExec
        let exec_expr = ExecExpr::new(ExecExprKind::new(BaseExec::CpuThread));
        let error = TyError::IllegalExec(exec_expr);

        match error {
            TyError::IllegalExec(_) => {}
            _ => panic!("Expected TyError::IllegalExec"),
        }

        // Test TyError::PatternAndTypeDoNotMatch
        let pattern = Pattern::Ident(Mutability::Mut, Ident::new("x"));
        let ty = Ty::new(TyKind::Data(Box::new(DataTy::new(DataTyKind::Scalar(
            ScalarTy::I32,
        )))));
        let error = TyError::PatternAndTypeDoNotMatch(pattern, ty);

        match error {
            TyError::PatternAndTypeDoNotMatch(_, _) => {}
            _ => panic!("Expected TyError::PatternAndTypeDoNotMatch"),
        }

        // Test TyError::PrvValueAlreadyInUse
        let error = TyError::PrvValueAlreadyInUse("r1".to_string());

        match error {
            TyError::PrvValueAlreadyInUse(prv) => {
                assert_eq!(prv, "r1");
            }
            _ => panic!("Expected TyError::PrvValueAlreadyInUse"),
        }

        // Test TyError::CannotInferGenericArg
        let ident = Ident::new("T");
        let error = TyError::CannotInferGenericArg(ident.clone());

        match error {
            TyError::CannotInferGenericArg(found_ident) => {
                assert_eq!(found_ident.name.as_ref(), "T");
            }
            _ => panic!("Expected TyError::CannotInferGenericArg"),
        }

        // Test TyError::UnsafeRequired
        let expr = Expr::new(ExprKind::Hole);
        let error = TyError::UnsafeRequired(expr);

        match error {
            TyError::UnsafeRequired(_) => {}
            _ => panic!("Expected TyError::UnsafeRequired"),
        }
    }

    #[test]
    fn test_error_with_spans() {
        use crate::ast::*;

        // Test error creation with spans
        let mut ident = Ident::new("x");
        ident.span = Some(Span { begin: 10, end: 15 });

        let error = TyError::CtxError(CtxError::IdentNotFound(ident));

        match error {
            TyError::CtxError(CtxError::IdentNotFound(found_ident)) => {
                assert_eq!(found_ident.name.as_ref(), "x");
                assert!(found_ident.span.is_some());
                if let Some(span) = found_ident.span {
                    assert_eq!(span.begin, 10);
                    assert_eq!(span.end, 15);
                }
            }
            _ => panic!("Expected CtxError::IdentNotFound"),
        }
    }

    #[test]
    fn test_multi_error_creation() {
        let place_expr1 = PlaceExpr::new(PlaceExprKind::Ident(Ident::new("x")));
        let error1 = TyError::ReferenceToDeadTy(place_expr1);
        let error2 = TyError::IllegalExec(ExecExpr::new(ExecExprKind::new(BaseExec::CpuThread)));
        let multi_error = TyError::MultiError(vec![error1, error2]);

        match multi_error {
            TyError::MultiError(errors) => {
                assert_eq!(errors.len(), 2);
                assert!(matches!(errors[0], TyError::ReferenceToDeadTy(_)));
                assert!(matches!(errors[1], TyError::IllegalExec(_)));
            }
            _ => panic!("Expected TyError::MultiError"),
        }
    }

    #[test]
    fn test_error_from_conversions() {
        // Test From<CtxError> for TyError
        let ctx_err = CtxError::IdentNotFound(Ident::new("x"));
        let ty_err: TyError = ctx_err.into();

        match ty_err {
            TyError::CtxError(CtxError::IdentNotFound(_)) => {}
            _ => panic!("Expected TyError::CtxError"),
        }

        // Test From<SubTyError> for TyError
        let subty_err = SubTyError::Dummy;
        let ty_err: TyError = subty_err.into();

        match ty_err {
            TyError::SubTyError(SubTyError::Dummy) => {}
            _ => panic!("Expected TyError::SubTyError"),
        }

        // Test From<UnifyError> for TyError
        let unify_err = UnifyError::InfiniteType;
        let ty_err: TyError = unify_err.into();

        match ty_err {
            TyError::UnifyError(UnifyError::InfiniteType) => {}
            _ => panic!("Expected TyError::UnifyError"),
        }
    }
}
