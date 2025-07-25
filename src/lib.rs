extern crate core;

use crate::error::ErrorReported;

mod ast;
mod codegen;
pub mod error;
pub mod parser;
pub mod ty_check;

pub enum Backend {
    Cuda,
    AscendC,
}

pub fn compile(file_path: &str, backend: Backend) -> Result<String, ErrorReported> {
    let source = parser::SourceCode::from_file(file_path)?;
    let mut compil_unit = parser::parse(&source)?;
    ty_check::ty_check(&mut compil_unit)?;
    match backend {
        Backend::Cuda => Ok(codegen::cuda::gen(&compil_unit, false)),
        Backend::AscendC => Ok(codegen::ascend::gen(&compil_unit, false)),
    }
}
