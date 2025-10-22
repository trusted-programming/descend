//! MLIR Codegen Error Handling
//!
//! This module provides comprehensive error handling for the MLIR codegen backend.
//! It defines custom error types that capture different failure modes during
//! MLIR code generation, providing detailed error information for debugging.
//!
//! # Error Types
//!
//! - `General`: For general MLIR codegen errors with custom messages
//! - `TypeParseError`: For MLIR type parsing failures
//! - `OperationBuildError`: For MLIR operation construction failures
//! - `MissingResult`: For operations that fail to produce expected results
//! - `InvalidOperation`: For operations that are malformed or invalid
//! - `ContextError`: For context-related errors during codegen
//!
//! # Usage
//!
//! ```rust
//! use descend::codegen::mlir::error::{MlirError, type_parse_error};
//!
//! // Create specific error types
//! let err = type_parse_error("invalid_type");
//!
//! // Convert to string for display
//! println!("Error: {}", err);
//! ```

use std::fmt;

/// Errors that can occur during MLIR code generation
#[derive(Debug, Clone)]
pub enum MlirError {
    /// Failed to parse an MLIR type
    TypeParseError(String),
    /// Failed to build an MLIR operation
    OperationBuildError(String),
    /// Invalid operation or missing required components
    InvalidOperation(String),
    /// Missing expected result from operation
    MissingResult(String),
    /// Context-related error
    ContextError(String),
    /// General MLIR error with context
    General(String),
}

impl fmt::Display for MlirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlirError::TypeParseError(msg) => write!(f, "Type parsing failed: {}", msg),
            MlirError::OperationBuildError(msg) => write!(f, "Operation building failed: {}", msg),
            MlirError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            MlirError::MissingResult(msg) => write!(f, "Missing operation result: {}", msg),
            MlirError::ContextError(msg) => write!(f, "Context error: {}", msg),
            MlirError::General(msg) => write!(f, "MLIR error: {}", msg),
        }
    }
}

impl std::error::Error for MlirError {}

/// Helper function to create a type parse error with context
pub fn type_parse_error(type_str: &str) -> MlirError {
    MlirError::TypeParseError(format!("Failed to parse type '{}'", type_str))
}

/// Helper function to create an operation build error with context
pub fn operation_build_error(operation: &str) -> MlirError {
    MlirError::OperationBuildError(format!("Failed to build operation '{}'", operation))
}

/// Helper function to create a missing result error with context
pub fn missing_result_error(operation: &str, index: usize) -> MlirError {
    MlirError::MissingResult(format!(
        "Operation '{}' missing result at index {}",
        operation, index
    ))
}

/// Helper function to create a context error with context
pub fn context_error(msg: &str) -> MlirError {
    MlirError::ContextError(msg.to_string())
}

/// Helper function to create a general error with context
pub fn general_error(msg: &str) -> MlirError {
    MlirError::General(msg.to_string())
}
