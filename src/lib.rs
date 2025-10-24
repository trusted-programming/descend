extern crate core;

use crate::error::{CodegenErrorData, CompileError};

mod ast;
pub mod codegen;
pub mod error;
pub mod parser;
pub mod ty_check;

pub fn compile(file_path: &str) -> Result<(String, String), CompileError> {
    let source = parser::SourceCode::from_file(file_path)?;
    let mut compil_unit = parser::parse(&source)?;
    ty_check::ty_check(&mut compil_unit)?;

    let code_string = codegen::mlir::gen_checked(&compil_unit, false).map_err(|e| {
        CompileError::Codegen(CodegenErrorData {
            message: format!("MLIR verification failed: {}", e),
            span: None, // TODO: Extract span from codegen error
        })
    })?;

    let ast_string = format!("{:#?}", compil_unit.items);

    Ok((code_string, ast_string))
}
