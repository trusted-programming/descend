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
    ir::{operation::OperationLike, Block, BlockLike, Module, RegionLike, Location},
    Context,
};

use super::to_mlir::ToMlir;
use crate::ast as desc;
use melior::ir::Type;
use std::collections::HashMap;

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

        // Create entry block with function parameter arguments
        let location = Location::unknown(self.context);
        let param_types: Vec<(Type<'_>, Location<'_>)> = fun
            .param_decls
            .iter()
            .filter_map(|p| p.ty.as_ref())
            .map(|ty| (ty.to_mlir(self.context), location))
            .collect();
        let entry_block = region.append_block(Block::new(&param_types));

        // For single-pass legacy path, we create an empty map
        let fn_results: HashMap<String, Vec<melior::ir::Type<'_>>> = HashMap::new();

        // Create context for code generation
        let mut mlir_ctx = MlirContext::new(self.context, entry_block, fn_results);

        // Bind function parameters to block arguments
        for (i, param) in fun.param_decls.iter().enumerate() {
            if let Some(arg) = mlir_ctx.current_block.argument(i).ok().map(|a| a.into()) {
                mlir_ctx
                    .variables
                    .insert(param.ident.name.to_string(), arg);
            }
        }

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

    /// Build all functions in two passes so that calls can reference result types
    pub fn build_items_two_pass(&mut self, comp: &desc::CompilUnit) {
        // Pass 1: declare all functions and record result types
        let mut results_map: HashMap<String, Vec<Type<'_>>> = HashMap::new();

        // Keep handles to the appended ops to reuse their regions
        let mut fun_ops = Vec::new();

        for item in &comp.items {
            if let desc::Item::FunDef(fun) = item {
                let op = fun.to_mlir(self.context);
                let op_ref = self.module.body().append_operation(op);

                // Derive result types directly from AST return data type
                let ret_ty = fun.ret_dty.to_mlir(self.context);
                let res_types: Vec<Type<'_>> = if ret_ty.to_string() == "none" { vec![] } else { vec![ret_ty] };
                results_map.insert(fun.ident.name.to_string(), res_types);

                fun_ops.push((fun.ident.name.to_string(), op_ref, fun));
            }
        }

        // Pass 2: build bodies
        for (_name, op_ref, fun) in fun_ops {
            let region = op_ref.region(0).expect("Function should have a region");
            // Create entry block with function parameter argument types
            let location = Location::unknown(self.context);
            let param_types: Vec<(Type<'_>, Location<'_>)> = fun
                .param_decls
                .iter()
                .filter_map(|p| p.ty.as_ref())
                .map(|ty| (ty.to_mlir(self.context), location))
                .collect();
            let entry_block = region.append_block(Block::new(&param_types));

            let mut ctx = MlirContext::new(self.context, entry_block, results_map.clone());

            // Bind function parameters to block arguments
            for (i, param) in fun.param_decls.iter().enumerate() {
                if let Some(arg) = ctx.current_block.argument(i).ok().map(|a| a.into()) {
                    ctx.variables.insert(param.ident.name.to_string(), arg);
                }
            }
            let result_value = build_expr(&fun.body.body, &mut ctx);
            let location = ctx.location();
            if let Some(value) = result_value {
                let return_op = func::r#return(&[value], location);
                entry_block.append_operation(return_op);
            } else {
                let return_op = func::r#return(&[], location);
                entry_block.append_operation(return_op);
            }
        }
    }
}
