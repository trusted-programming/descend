pub mod builder;
pub mod dialects; // Generated dialect bindings
pub mod to_mlir;

use builder::MlirBuilder;
use melior::{
    dialect::DialectRegistry,
    ir::{operation::OperationLike, Location, Module},
    utility::register_all_dialects,
    Context,
};

use crate::ast::CompilUnit;

pub fn gen(comp_unit: &CompilUnit, _idx_checks: bool) -> String {
    let context = create_context();
    let location = Location::unknown(&context);
    let module = Module::new(location);
    let mut builder = MlirBuilder::new(&context, module);

    // Two-pass build so that calls know callee result types
    builder.build_items_two_pass(comp_unit);

    // Dump the module to string
    builder.module().as_operation().to_string()
}

pub fn gen_checked(comp_unit: &CompilUnit, _idx_checks: bool) -> Result<String, String> {
    let context = create_context();
    let location = Location::unknown(&context);
    let module = Module::new(location);
    let mut builder = MlirBuilder::new(&context, module);

    // Two-pass build so that calls know callee result types
    builder.build_items_two_pass(comp_unit);

    // Verify the module
    if !builder.module().as_operation().verify() {
        return Err("MLIR module verification failed".to_string());
    }

    // Dump the module to string
    Ok(builder.module().as_operation().to_string())
}

pub fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    // Custom dialects (hivm, annotation, symbol) are loaded via dialects.rs module

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
}
