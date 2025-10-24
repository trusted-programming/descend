use crate::parser::SourceCode;
use miette::{Diagnostic, LabeledSpan, SourceSpan};
use std::fmt::Formatter;
use thiserror::Error;

#[must_use]
#[derive(Debug, Diagnostic)]
#[diagnostic(severity(Error), code("file_io_error"))]
pub struct FileIOError<'a> {
    file_path: &'a str,
    io_error: std::io::Error,
}

impl<'a> FileIOError<'a> {
    pub fn new(file_path: &'a str, io_error: std::io::Error) -> Self {
        FileIOError {
            file_path,
            io_error,
        }
    }

    pub fn emit(&self) -> ErrorReported {
        eprintln!("couldn't read {}: {}", self.file_path, self.io_error);
        ErrorReported
    }

    /// Convert to CompileError for better error handling
    pub fn to_compile_error(self) -> CompileError {
        CompileError::FileIO(FileIOErrorData {
            file_path: self.file_path.to_string(),
            io_error_kind: self.io_error.kind(),
            error_message: error_kind_to_message(self.io_error.kind()),
            span: None, // No specific span for file IO errors
        })
    }
}

impl<'a> std::fmt::Display for FileIOError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "couldn't read {}: {}", self.file_path, self.io_error)
    }
}

impl<'a> std::error::Error for FileIOError<'a> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.io_error)
    }
}

pub fn single_line_snippet<'a>(
    source: &'a SourceCode<'a>,
    _label: &'a str,
    annotation: &'a str,
    line_num: u32,
    begin_column: u32,
    end_column: u32,
) -> miette::Report {
    if let Some(_line) = source.get_line(line_num) {
        let start_offset = source.get_offset(line_num, begin_column);
        let end_offset = source.get_offset(line_num, end_column);

        miette::miette!(
            labels = [LabeledSpan::new_with_span(
                Some(annotation.to_string()),
                start_offset..end_offset
            )],
            "{}",
            annotation
        )
    } else {
        miette::miette!("{}", annotation)
    }
}

/// Create a multi-span error report with primary and secondary labels
pub fn multi_span_snippet<'a>(
    source: &'a SourceCode<'a>,
    message: &'a str,
    primary_span: Option<(u32, u32, u32, &'a str)>, // (line, begin_col, end_col, label)
    secondary_spans: Vec<(u32, u32, u32, &'a str)>, // Vec of (line, begin_col, end_col, label)
) -> miette::Report {
    let mut labels = Vec::new();

    // Add primary span
    if let Some((line, begin_col, end_col, label)) = primary_span {
        if let Some(_line) = source.get_line(line) {
            let start_offset = source.get_offset(line, begin_col);
            let end_offset = source.get_offset(line, end_col);
            labels.push(LabeledSpan::new_with_span(
                Some(label.to_string()),
                start_offset..end_offset,
            ));
        }
    }

    // Add secondary spans
    for (line, begin_col, end_col, label) in secondary_spans {
        if let Some(_line) = source.get_line(line) {
            let start_offset = source.get_offset(line, begin_col);
            let end_offset = source.get_offset(line, end_col);
            labels.push(LabeledSpan::new_with_span(
                Some(label.to_string()),
                start_offset..end_offset,
            ));
        }
    }

    miette::miette!(labels = labels, "{}", message)
}

/// Add help text to an existing report
pub fn with_help(report: miette::Report, _help_text: &str) -> miette::Report {
    // For now, just return the original report since miette macro syntax is complex
    // TODO: Implement proper help text addition
    report
}

/// Add related information to an existing report
pub fn with_related(report: miette::Report, _related_text: &str) -> miette::Report {
    // For now, just return the original report since miette macro syntax is complex
    // TODO: Implement proper related text addition
    report
}

/// Enum representing different types of compilation errors
#[derive(Error, Diagnostic, Debug, Clone)]
pub enum CompileError {
    #[error(transparent)]
    #[diagnostic(
        code(descend::file_io_error),
        help("Check that the file exists and you have permission to read it")
    )]
    FileIO(#[from] FileIOErrorData),

    #[error(transparent)]
    #[diagnostic(
        code(descend::parse_error),
        help("Check the syntax of your Descend code")
    )]
    Parse(#[from] ParseErrorData),

    #[error(transparent)]
    #[diagnostic(
        code(descend::type_check_error),
        help("Check your type annotations and variable usage")
    )]
    TypeCheck(#[from] TypeCheckErrorData),

    #[error(transparent)]
    #[diagnostic(
        code(descend::codegen_error),
        help("This is an internal compiler error. Please report it as a bug")
    )]
    Codegen(#[from] CodegenErrorData),
}

/// Data for FileIO errors with miette diagnostic support
#[derive(Error, Diagnostic, Debug, Clone)]
#[error("Failed to read file '{file_path}': {error_message}")]
#[diagnostic(
    code(descend::file_io_error),
    help("Check that the file exists and you have permission to read it")
)]
pub struct FileIOErrorData {
    pub file_path: String,
    pub io_error_kind: std::io::ErrorKind,
    pub error_message: String,

    #[label("File that could not be read")]
    pub span: Option<SourceSpan>,
}

/// Data for Parse errors with miette diagnostic support
#[derive(Error, Diagnostic, Debug, Clone)]
#[error("Parse error: {message}")]
#[diagnostic(
    code(descend::parse_error),
    help("Check the syntax of your Descend code")
)]
pub struct ParseErrorData {
    pub message: String,

    #[label("Parse error location")]
    pub span: Option<SourceSpan>,
}

/// Data for TypeCheck errors with miette diagnostic support
#[derive(Error, Diagnostic, Debug, Clone)]
#[error("Type check error: {message}")]
#[diagnostic(
    code(descend::type_check_error),
    help("Check your type annotations and variable usage")
)]
pub struct TypeCheckErrorData {
    pub message: String,

    #[label("Type error location")]
    pub span: Option<SourceSpan>,
    
    /// Additional labeled spans for complex errors (e.g., conflicting borrows)
    pub secondary_spans: Vec<(SourceSpan, String)>,
    
    /// Additional help text
    pub help_text: Option<String>,
    
    /// Related information
    pub related: Option<String>,
}

/// Data for Codegen errors with miette diagnostic support
#[derive(Error, Diagnostic, Debug, Clone)]
#[error("Code generation error: {message}")]
#[diagnostic(
    code(descend::codegen_error),
    help("This is an internal compiler error. Please report it as a bug")
)]
pub struct CodegenErrorData {
    pub message: String,

    #[label("Code generation error location")]
    pub span: Option<SourceSpan>,
}

impl CompileError {
    /// Attach source code to the error for beautiful display
    pub fn with_source_code(self, source: miette::NamedSource<String>) -> miette::Report {
        miette::Report::new(self).with_source_code(source)
    }
}

/// Convert ErrorKind to human-readable message
fn error_kind_to_message(kind: std::io::ErrorKind) -> String {
    match kind {
        std::io::ErrorKind::NotFound => "entity not found".to_string(),
        std::io::ErrorKind::PermissionDenied => "permission denied".to_string(),
        std::io::ErrorKind::InvalidInput => "invalid input".to_string(),
        std::io::ErrorKind::UnexpectedEof => "unexpected end of file".to_string(),
        _ => "unknown error".to_string(),
    }
}

/// Convert std::io::Error to CompileError for easy error handling
impl From<std::io::Error> for CompileError {
    fn from(error: std::io::Error) -> Self {
        CompileError::FileIO(FileIOErrorData {
            file_path: "unknown file".to_string(),
            io_error_kind: error.kind(),
            error_message: error_kind_to_message(error.kind()),
            span: None,
        })
    }
}

/// Convert TyError to CompileError with span preservation
impl From<crate::ty_check::error::TyError> for CompileError {
    fn from(error: crate::ty_check::error::TyError) -> Self {
        use crate::ty_check::error::TyError;
        
        match error {
            TyError::MultiError(errors) => {
                // For now, just take the first error. In the future, we could collect all errors
                if let Some(first_error) = errors.into_iter().next() {
                    first_error.into()
                } else {
                    CompileError::TypeCheck(TypeCheckErrorData {
                        message: "Multiple type checking errors occurred".to_string(),
                        span: None,
                        secondary_spans: vec![],
                        help_text: Some("Fix each error individually".to_string()),
                        related: None,
                    })
                }
            }
            TyError::MutabilityNotAllowed(ty) => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: "Mutability not allowed".to_string(),
                    span: ty.span.map(|s| s.into()),
                    secondary_spans: vec![],
                    help_text: Some("This type does not allow mutability. Consider using a mutable type or removing the mutability requirement.".to_string()),
                    related: None,
                })
            }
            TyError::MismatchedDataTypes(expected, actual, expr) => {
                use crate::ast::printer::PrintState;
                let mut expected_printer = PrintState::new();
                expected_printer.print_dty(&expected);
                let mut actual_printer = PrintState::new();
                actual_printer.print_dty(&actual);
                
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: format!("Expected `{}` but found `{}`", expected_printer.get(), actual_printer.get()),
                    span: expr.span.map(|s| s.into()),
                    secondary_spans: vec![],
                    help_text: Some("The types do not match. Check that you're using the correct types for this operation.".to_string()),
                    related: None,
                })
            }
            TyError::ConflictingBorrow(place_expr, ownership, conflict) => {
                let mut secondary_spans = vec![];
                let mut help_text = "This borrow conflicts with an existing borrow. Consider restructuring your code to avoid simultaneous borrows.".to_string();
                
                // Add secondary span for the conflicting location
                if let Some(place_span) = place_expr.span {
                    secondary_spans.push((place_span.into(), "conflicting borrow".to_string()));
                }
                
                // Add ownership-specific help
                help_text.push_str(&format!(" Ownership: {:?}.", ownership));
                
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: "Conflicting borrow detected".to_string(),
                    span: place_expr.span.map(|s| s.into()),
                    secondary_spans,
                    help_text: Some(help_text),
                    related: Some(format!("Conflict details: {:?}", conflict)),
                })
            }
            TyError::ReferenceToDeadTy(place_expr) => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: "Reference points to moved value".to_string(),
                    span: place_expr.span.map(|s| s.into()),
                    secondary_spans: vec![],
                    help_text: Some("The value this reference points to has been moved. Consider restructuring your code to avoid moving values that are still referenced.".to_string()),
                    related: None,
                })
            }
            TyError::IllegalExec(exec_expr) => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: "Illegal execution context".to_string(),
                    span: exec_expr.span.map(|s| s.into()),
                    secondary_spans: vec![],
                    help_text: Some("The execution context is not valid for this operation. Check that you're using the correct execution resource (CPU thread, NPU block, etc.).".to_string()),
                    related: None,
                })
            }
            TyError::AssignToConst(place_expr) => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: "Cannot assign to constant".to_string(),
                    span: place_expr.span.map(|s| s.into()),
                    secondary_spans: vec![],
                    help_text: Some("Make the variable mutable by declaring it with 'mut' to allow assignment.".to_string()),
                    related: None,
                })
            }
            TyError::ConstBorrow(place_expr) => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: "Cannot borrow as unique".to_string(),
                    span: place_expr.span.map(|s| s.into()),
                    secondary_spans: vec![],
                    help_text: Some("Make the variable mutable by declaring it with 'mut' to allow unique borrowing.".to_string()),
                    related: None,
                })
            }
            TyError::ExpectedTupleType(ty_kind, place_expr) => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: format!("Expected tuple type but found `{:?}`", ty_kind),
                    span: place_expr.span.map(|s| s.into()),
                    secondary_spans: vec![],
                    help_text: Some("This operation expects a tuple type. Check that you're using the correct type.".to_string()),
                    related: None,
                })
            }
            TyError::UnifyError(unify_err) => {
                match unify_err {
                    crate::ty_check::error::UnifyError::CannotUnify { left_type, right_type, context, span } => {
                        use crate::ty_check::error::UnifyContext;
                        use crate::ast::printer::PrintState;
                        use crate::ast::TyKind;
                        
                        // Pretty print the types
                        let mut left_printer = PrintState::new();
                        if let TyKind::Data(dty) = &left_type.ty {
                            left_printer.print_dty(dty);
                        }
                        let mut right_printer = PrintState::new();
                        if let TyKind::Data(dty) = &right_type.ty {
                            right_printer.print_dty(dty);
                        }
                        
                        // Format context description
                        let context_desc = match context {
                            UnifyContext::FunctionParameter(idx) => format!("function parameter {}", idx + 1),
                            UnifyContext::FunctionReturn => "function return type".to_string(),
                            UnifyContext::VariableAssignment(ident) => format!("assignment to variable '{}'", ident.name),
                            UnifyContext::ArrayElement => "array element type".to_string(),
                            UnifyContext::TupleElement(idx) => format!("tuple element {}", idx + 1),
                            UnifyContext::StructField(ident) => format!("struct field '{}'", ident.name),
                            UnifyContext::PatternMatch => "pattern matching".to_string(),
                            UnifyContext::GenericParameter(ident) => format!("generic parameter '{}'", ident.name),
                            UnifyContext::Expression(_) => "expression".to_string(),
                            UnifyContext::Other(desc) => desc.clone(),
                        };
                        
                        CompileError::TypeCheck(TypeCheckErrorData {
                            message: format!("Expected `{}` but found `{}` in {}", right_printer.get(), left_printer.get(), context_desc),
                            span: span.map(|s| s.into()),
                            secondary_spans: vec![],
                            help_text: Some("The types cannot be unified in this context. Check your type annotations.".to_string()),
                            related: None,
                        })
                    }
                    crate::ty_check::error::UnifyError::InfiniteType => {
                        CompileError::TypeCheck(TypeCheckErrorData {
                            message: "Infinite type detected during unification".to_string(),
                            span: None,
                            secondary_spans: vec![],
                            help_text: Some("A type variable is being unified with a term that contains itself, creating an infinite type.".to_string()),
                            related: None,
                        })
                    }
                    crate::ty_check::error::UnifyError::SubTyError(sub_err) => {
                        CompileError::TypeCheck(TypeCheckErrorData {
                            message: "Subtyping error".to_string(),
                            span: None,
                            secondary_spans: vec![],
                            help_text: Some("The types do not satisfy the subtyping relationship.".to_string()),
                            related: Some(format!("Subtyping error: {:?}", sub_err)),
                        })
                    }
                }
            }
            TyError::CtxError(ctx_err) => {
                match ctx_err {
                    crate::ty_check::error::CtxError::IdentNotFound(ident) => {
                        CompileError::TypeCheck(TypeCheckErrorData {
                            message: format!("Identifier '{}' not found in context", ident.name),
                            span: ident.span.map(|s| s.into()),
                            secondary_spans: vec![],
                            help_text: Some("Check that the identifier is declared before use, or consider if there's a typo in the name.".to_string()),
                            related: None,
                        })
                    }
                    crate::ty_check::error::CtxError::KindedIdentNotFound(ident) => {
                        CompileError::TypeCheck(TypeCheckErrorData {
                            message: format!("Identifier '{}' not found in kinding context", ident.name),
                            span: ident.span.map(|s| s.into()),
                            secondary_spans: vec![],
                            help_text: Some("This identifier needs to be declared with a kind (type, natural number, memory, or provenance) before use.".to_string()),
                            related: None,
                        })
                    }
                    crate::ty_check::error::CtxError::PrvValueNotFound(prv_name) => {
                        CompileError::TypeCheck(TypeCheckErrorData {
                            message: format!("Provenance value '{}' not found in typing context", prv_name),
                            span: None,
                            secondary_spans: vec![],
                            help_text: Some("Ensure the provenance is properly declared and available in the current scope.".to_string()),
                            related: None,
                        })
                    }
                    crate::ty_check::error::CtxError::PrvIdentNotFound(ident) => {
                        CompileError::TypeCheck(TypeCheckErrorData {
                            message: format!("Provenance identifier '{}' is not declared", ident.name),
                            span: ident.span.map(|s| s.into()),
                            secondary_spans: vec![],
                            help_text: Some("Declare this provenance identifier before use, or check for typos in the name.".to_string()),
                            related: None,
                        })
                    }
                    crate::ty_check::error::CtxError::OutlRelNotDefined(longer, shorter) => {
                        let mut secondary_spans = vec![];
                        if let Some(span) = longer.span {
                            secondary_spans.push((span.into(), "longer lifetime".to_string()));
                        }
                        if let Some(span) = shorter.span {
                            secondary_spans.push((span.into(), "shorter lifetime".to_string()));
                        }
                        
                        CompileError::TypeCheck(TypeCheckErrorData {
                            message: "Outlives relation not defined".to_string(),
                            span: None,
                            secondary_spans,
                            help_text: Some(format!("Define the outlives relation: {} outlives {}", longer.name, shorter.name)),
                            related: None,
                        })
                    }
                    crate::ty_check::error::CtxError::IllegalProjection => {
                        CompileError::TypeCheck(TypeCheckErrorData {
                            message: "Illegal projection operation".to_string(),
                            span: None,
                            secondary_spans: vec![],
                            help_text: Some("This projection operation is not allowed in the current context. Check the type structure and projection rules.".to_string()),
                            related: None,
                        })
                    }
                }
            }
            TyError::MissingMain => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: "Missing main function".to_string(),
                    span: None,
                    secondary_spans: vec![],
                    help_text: Some("A main function is required as the entry point of your program.".to_string()),
                    related: None,
                })
            }
            TyError::NatEvalError(nat_err, span) => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: "Natural number evaluation error".to_string(),
                    span: span.map(|s| s.into()),
                    secondary_spans: vec![],
                    help_text: Some("There was an error evaluating a natural number expression.".to_string()),
                    related: Some(format!("Natural number error: {:?}", nat_err)),
                })
            }
            TyError::CannotInferGenericArg(ident) => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: format!("Cannot infer generic argument for '{}'", ident.name),
                    span: ident.span.map(|s| s.into()),
                    secondary_spans: vec![],
                    help_text: Some("Provide explicit type annotations for generic arguments or ensure the context provides enough information for inference.".to_string()),
                    related: None,
                })
            }
            TyError::UnsafeRequired(expr) => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: "Unsafe operation required".to_string(),
                    span: expr.span.map(|s| s.into()),
                    secondary_spans: vec![],
                    help_text: Some("This operation requires unsafe code. Wrap it in an 'unsafe' block if you're certain it's safe.".to_string()),
                    related: None,
                })
            }
            TyError::String(msg) => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: msg,
                    span: None,
                    secondary_spans: vec![],
                    help_text: None,
                    related: None,
                })
            }
            // Add more variants as needed...
            _ => {
                CompileError::TypeCheck(TypeCheckErrorData {
                    message: format!("Type check error: {:?}", error),
                    span: None,
                    secondary_spans: vec![],
                    help_text: Some("Check your type annotations and variable usage.".to_string()),
                    related: None,
                })
            }
        }
    }
}

pub struct ErrorReported;

impl std::fmt::Debug for ErrorReported {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Aborting due to previous error.")
    }
}

impl std::fmt::Display for ErrorReported {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Aborting due to previous error.")
    }
}
