extern crate core;

use crate::error::ErrorReported;

mod ast;
mod codegen;
pub mod error;
pub mod parser;
pub mod ty_check;

pub enum Backend {
    Cuda,
    Mlir,
}

pub fn compile(file_path: &str, backend: Backend) -> Result<(String, String), ErrorReported> {
    let source = parser::SourceCode::from_file(file_path)?;
    let mut compil_unit = parser::parse(&source)?;

    ty_check::ty_check(&mut compil_unit)?;

    let code_string = match backend {
        Backend::Cuda => codegen::cuda::gen(&compil_unit, false),
        Backend::Mlir => codegen::mlir::gen_checked(&compil_unit, false).map_err(|e| {
            eprintln!("MLIR verification failed: {}", e);
            ErrorReported
        })?,
    };
    let ast_string = format!("{:#?}", compil_unit.items);

    Ok((code_string, ast_string))
}

pub fn compile_unchecked(file_path: &str, backend: Backend) -> Result<String, ErrorReported> {
    let source = parser::SourceCode::from_file(file_path)?;
    let mut compil_unit = parser::parse(&source)?;

    ty_check::ty_check(&mut compil_unit)?;

    let code_string = match backend {
        Backend::Cuda => codegen::cuda::gen(&compil_unit, false),
        Backend::Mlir => codegen::mlir::gen(&compil_unit, false),
    };

    Ok(code_string)
}
