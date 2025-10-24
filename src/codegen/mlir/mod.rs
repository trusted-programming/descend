pub mod builder;
pub mod error;
pub mod to_mlir;

use builder::MlirBuilder;
use error::MlirError;
use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{Location, Module, operation::OperationLike},
    utility::register_all_dialects,
};

use crate::ast::CompilUnit;

/// Internal helper function to build MLIR module
fn build_module_internal(comp_unit: &CompilUnit) -> Result<String, MlirError> {
    // Check if we need HIVM address spaces
    if needs_hivm_address_space(comp_unit) {
        // For HIVM, we use string generation instead of proper module building
        // This is a limitation of the current implementation
        return Err(MlirError::General(
            "HIVM address spaces require string generation path".to_string(),
        ));
    }

    let context = create_context();
    let location = Location::unknown(&context);
    let module = Module::new(location);
    let mut builder = MlirBuilder::new(&context, module);

    // Two-pass build so that calls know callee result types
    builder.build_items_two_pass(comp_unit);

    // Verify the module before generating the string
    if !builder.module().as_operation().verify() {
        return Err(MlirError::General(
            "MLIR module verification failed".to_string(),
        ));
    }

    Ok(builder.module().as_operation().to_string())
}

pub fn gen_checked(comp_unit: &CompilUnit, _idx_checks: bool) -> Result<String, String> {
    // Check if we need HIVM address spaces
    if needs_hivm_address_space(comp_unit) {
        Ok(to_mlir::types::generate_mlir_string_with_hivm(comp_unit))
    } else {
        match build_module_internal(comp_unit) {
            Ok(mlir_string) => {
                // Note: We can't verify the module here since we only have the string
                // The verification would need to be done during the building process
                Ok(mlir_string)
            }
            Err(e) => Err(format!("MLIR module building failed: {}", e)),
        }
    }
}

/// Check if the compilation unit needs HIVM address spaces
fn needs_hivm_address_space(comp_unit: &CompilUnit) -> bool {
    for item in &comp_unit.items {
        if let crate::ast::Item::FunDef(fun) = item {
            for param in &fun.param_decls {
                if let Some(_ty) = &param.ty {
                    // TODO: check if the type is a npu function
                }
            }
        }
    }
    false
}

pub fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();

    // Allow unregistered dialects to handle HIVM dialect
    context.set_allow_unregistered_dialects(true);

    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    context
}
