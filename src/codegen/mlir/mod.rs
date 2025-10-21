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
    // Check if we need HIVM address spaces
    if needs_hivm_address_space(comp_unit) {
        to_mlir::types::generate_mlir_string_with_hivm(comp_unit)
    } else {
        let context = create_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);
        let mut builder = MlirBuilder::new(&context, module);

        // Two-pass build so that calls know callee result types
        builder.build_items_two_pass(comp_unit);

        // Dump the module to string
        builder.module().as_operation().to_string()
    }
}

pub fn gen_checked(comp_unit: &CompilUnit, _idx_checks: bool) -> Result<String, String> {
    // Check if we need HIVM address spaces
    if needs_hivm_address_space(comp_unit) {
        Ok(to_mlir::types::generate_mlir_string_with_hivm(comp_unit))
    } else {
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
}

/// Check if the compilation unit needs HIVM address spaces
fn needs_hivm_address_space(comp_unit: &CompilUnit) -> bool {
    for item in &comp_unit.items {
        if let crate::ast::Item::FunDef(fun) = item {
            // Only check the main function or functions that are not HIVM placeholders
            if fun.ident.name == "main".into() || !is_hivm_placeholder_function(fun) {
                for param in &fun.param_decls {
                    if let Some(ty) = &param.ty {
                        if has_gpu_memory(ty) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Check if a function is a HIVM placeholder function
fn is_hivm_placeholder_function(fun: &crate::ast::FunDef) -> bool {
    fun.ident.name.starts_with("hivm_")
}

/// Check if a type has GPU memory qualifiers
fn has_gpu_memory(ty: &crate::ast::Ty) -> bool {
    match &ty.ty {
        crate::ast::TyKind::Data(data_ty) => {
            match &data_ty.dty {
                crate::ast::DataTyKind::At(_, mem) => {
                    matches!(mem, crate::ast::Memory::GpuGlobal | crate::ast::Memory::GpuShared | crate::ast::Memory::GpuLocal)
                },
                crate::ast::DataTyKind::Ref(ref_dty) => {
                    matches!(ref_dty.mem, crate::ast::Memory::GpuGlobal | crate::ast::Memory::GpuShared | crate::ast::Memory::GpuLocal)
                },
                _ => false,
            }
        },
        _ => false,
    }
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

