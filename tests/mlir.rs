#[path = "mlir/core.rs"]
mod core;

#[path = "mlir/error_examples.rs"]
mod error_examples;

const BACKEND: descend::Backend = descend::Backend::Mlir;
