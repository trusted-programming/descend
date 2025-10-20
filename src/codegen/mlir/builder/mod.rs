pub mod context;
pub mod control_flow;
pub mod expr;
pub mod literal;
pub mod loops;
pub mod nat;
pub mod ops;
pub mod place;

use melior::{
    dialect::func,
    ir::{operation::OperationLike, Block, BlockLike, Module, RegionLike},
    Context,
};

use super::to_mlir::ToMlir;
use crate::ast as desc;

// Re-export the main types and functions
pub use context::MlirContext;
pub use expr::build_expr;

/// MLIR code generator using the builder pattern
pub struct MlirBuilder<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
}

impl<'ctx> MlirBuilder<'ctx> {
    pub fn new(context: &'ctx Context, module: Module<'ctx>) -> Self {
        Self { context, module }
    }

    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Build an item (top-level definition)
    pub fn build_item(&mut self, item: &desc::Item) {
        match item {
            desc::Item::FunDef(fun) => self.build_function(fun),
            desc::Item::FunDecl(_) => {
                // Function declarations don't have bodies, nothing to generate
            }
            desc::Item::StructDecl(_) => {
                // Struct declarations are handled at the type level
            }
        }
    }

    /// Build a function definition
    fn build_function(&mut self, fun: &desc::FunDef) {
        let function_op = fun.to_mlir(self.context);

        // Append and get reference to the function operation, then get its region
        let func_op_ref = self.module.body().append_operation(function_op);
        let region = func_op_ref
            .region(0)
            .expect("Function should have a region");
        let entry_block = region.append_block(Block::new(&[]));

        // Create context for code generation
        let mut mlir_ctx = MlirContext::new(self.context, entry_block);

        // Build the function body expression using the context
        let result_value = build_expr(&fun.body.body, &mut mlir_ctx);

        // Add return statement using the result value
        let location = mlir_ctx.location();
        if let Some(value) = result_value {
            let return_op = func::r#return(&[value], location);
            entry_block.append_operation(return_op);
        } else {
            // Return with no value (for unit/void functions)
            let return_op = func::r#return(&[], location);
            entry_block.append_operation(return_op);
        }
    }
}
