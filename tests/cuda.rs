#[path = "cuda/infer.rs"]
mod infer;

#[path = "cuda/rodinia.rs"]
mod rodinia;

#[path = "cuda/error_examples.rs"]
mod error_examples;

#[path = "cuda/simple.rs"]
mod simple;

const BACKEND: descend::Backend = descend::Backend::Cuda;
