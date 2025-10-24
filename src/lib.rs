extern crate core;

use crate::error::{CodegenErrorData, CompileError};

mod ast;
pub mod codegen;
pub mod error;
pub mod parser;
pub mod ty_check;

pub fn compile(file_path: &str) -> Result<(String, String), CompileError> {
    let source = parser::SourceCode::from_file(file_path)?;
    compile_from_source(&source)
}

pub fn compile_from_source(source: &parser::SourceCode) -> Result<(String, String), CompileError> {
    let mut compil_unit = parser::parse(source)?;
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

/// Compile a file and return a miette::Report for errors (with source code attached)
pub fn compile_with_source(file_path: &str) -> Result<(String, String), miette::Report> {
    let source_code = std::fs::read_to_string(file_path).map_err(|e| {
        use crate::error::{CompileError, FileIOErrorData};
        CompileError::FileIO(FileIOErrorData {
            file_path: file_path.to_string(),
            io_error_kind: e.kind(),
            error_message: crate::error::error_kind_to_message(e.kind()),
            span: None,
        })
        .with_source_code(miette::NamedSource::new(file_path, String::new()))
    })?;

    let named_source = miette::NamedSource::new(file_path, source_code.clone());
    let source = parser::SourceCode::from_file(file_path)
        .map_err(|e| e.with_source_code(named_source.clone()))?;

    match compile_from_source(&source) {
        Ok(result) => Ok(result),
        Err(err) => Err(err.with_source_code(named_source)),
    }
}
