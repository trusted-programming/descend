//! MLIR Code Generation Module
//!
//! This module provides code generation from the Descend AST to MLIR (Multi-Level Intermediate Representation).
//! It supports both standard MLIR generation and HIVM (Heterogeneous Intermediate Virtual Machine) dialect
//! with GPU memory address spaces.
//!
//! ## Architecture
//!
//! The codegen uses a two-pass compilation strategy:
//! 1. **Pass 1**: Declare all functions and record their result types
//! 2. **Pass 2**: Build function bodies with knowledge of callee result types
//!
//! This approach enables proper function call generation where callers know the return types
//! of their callees, which is essential for MLIR's SSA form.
//!
//! ## HIVM Address Spaces
//!
//! When GPU memory qualifiers are detected (GpuGlobal, GpuShared, GpuLocal), the codegen
//! switches to a string-based generation path that includes HIVM dialect address space attributes:
//! - `#hivm.address_space<gm>` for global memory
//! - `#hivm.address_space<ub>` for local memory
//!
//! ## Usage
//!
//! ```rust,no_run
//! // Example usage (requires proper compilation unit)
//! // let mlir_string = gen(&comp_unit, false);
//! // let mlir_string = gen_checked(&comp_unit, false)?;
//! ```

pub mod builder;
pub mod error;
pub mod to_mlir;

use builder::MlirBuilder;
use error::MlirError;
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module, operation::OperationLike},
    utility::register_all_dialects,
    Context,
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

pub fn gen(comp_unit: &CompilUnit, _idx_checks: bool) -> String {
    // Check if we need HIVM address spaces
    if needs_hivm_address_space(comp_unit) {
        to_mlir::types::generate_mlir_string_with_hivm(comp_unit)
    } else {
        // Use internal helper, but handle errors by falling back to string generation
        match build_module_internal(comp_unit) {
            Ok(mlir_string) => mlir_string,
            Err(_) => {
                // Fallback to string generation if internal building fails
                to_mlir::types::generate_mlir_string_with_hivm(comp_unit)
            }
        }
    }
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
        crate::ast::TyKind::Data(data_ty) => match &data_ty.dty {
            crate::ast::DataTyKind::At(_, mem) => {
                matches!(
                    mem,
                    crate::ast::Memory::GpuGlobal
                        | crate::ast::Memory::GpuShared
                        | crate::ast::Memory::GpuLocal
                )
            }
            crate::ast::DataTyKind::Ref(ref_dty) => {
                matches!(
                    ref_dty.mem,
                    crate::ast::Memory::GpuGlobal
                        | crate::ast::Memory::GpuShared
                        | crate::ast::Memory::GpuLocal
                )
            }
            _ => false,
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
