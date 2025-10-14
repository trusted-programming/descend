use melior::{
    dialect::{arith, func},
    ir::{
        attribute::IntegerAttribute, operation::OperationLike, r#type::IntegerType, Block,
        BlockLike, BlockRef, Location, Module, RegionLike, Value,
    },
    Context,
};
use std::collections::HashMap;

use super::to_mlir::ToMlir;
use crate::ast as desc;
use desc::BinOp;

/// Context for MLIR code generation with state management
pub struct MlirContext<'ctx, 'a, 'b> {
    pub context: &'ctx Context,
    pub variables: HashMap<String, Value<'a, 'b>>,
    pub current_block: BlockRef<'a, 'b>,
}

impl<'ctx, 'a, 'b> MlirContext<'ctx, 'a, 'b> {
    pub fn new(context: &'ctx Context, block: BlockRef<'a, 'b>) -> Self {
        Self {
            context,
            variables: HashMap::new(),
            current_block: block,
        }
    }

    pub fn location(&self) -> Location<'ctx> {
        Location::unknown(self.context)
    }
}

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
            _ => panic!("Unhandled item type"),
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

/// Build an expression using the MlirContext
fn build_expr<'ctx, 'a, 'b>(
    expr: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    use desc::ExprKind;

    match &expr.expr {
        ExprKind::Hole => None,
        ExprKind::Lit(lit) => build_literal(lit, ctx),
        ExprKind::Block(block_expr) => {
            // TODO: Process block.prvs if needed
            build_expr(&block_expr.body, ctx)
        }
        ExprKind::BinOp(op, lhs, rhs) => {
            // Build left and right operands
            let lhs_value = build_expr(lhs, ctx)?;
            let rhs_value = build_expr(rhs, ctx)?;

            // Create the appropriate arithmetic operation
            let location = ctx.location();
            let result_op = match op {
                BinOp::Add => arith::addi(lhs_value, rhs_value, location),
                BinOp::Sub => arith::subi(lhs_value, rhs_value, location),
                BinOp::Mul => arith::muli(lhs_value, rhs_value, location),
                BinOp::Div => arith::divsi(lhs_value, rhs_value, location),
                BinOp::Mod => arith::remsi(lhs_value, rhs_value, location),
                _ => unimplemented!("Unsupported binary operation: {:?}", op),
            };

            let op_ref = ctx.current_block.append_operation(result_op);
            Some(op_ref.result(0).unwrap().into())
        }
        ExprKind::Let(pattern, _ty, value_expr) => {
            use desc::Pattern;

            // Evaluate the value expression
            let value = build_expr(value_expr, ctx)?;

            // Bind the variable name to the SSA value
            match pattern {
                Pattern::Ident(_mutability, ident) => {
                    ctx.variables.insert(ident.name.to_string(), value);
                }
                _ => unimplemented!("Unsupported pattern in let binding: {:?}", pattern),
            }

            // Let expressions don't produce a value
            None
        }
        ExprKind::PlaceExpr(place_expr) => build_place_expr(place_expr, ctx),
        ExprKind::Seq(exprs) => {
            let mut last_value = None;
            for expr in exprs {
                last_value = build_expr(expr, ctx);
            }
            last_value
        }
        _ => unimplemented!("Unhandled expression type: {:?}", expr.expr),
    }
}

/// Build a literal constant
fn build_literal<'ctx, 'a, 'b>(
    lit: &desc::Lit,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    let location = ctx.location();

    match lit {
        desc::Lit::I32(value) => {
            let i32_type = IntegerType::new(ctx.context, 32).into();
            let value_attr = IntegerAttribute::new(i32_type, *value as i64).into();
            let const_op = arith::constant(ctx.context, value_attr, location);

            let op_ref = ctx.current_block.append_operation(const_op);
            Some(op_ref.result(0).unwrap().into())
        }
        _ => unimplemented!("Unsupported literal type: {:?}", lit),
    }
}

/// Build a place expression (variable lookup)
fn build_place_expr<'ctx, 'a, 'b>(
    place_expr: &desc::PlaceExpr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    use desc::PlaceExprKind;

    match &place_expr.pl_expr {
        PlaceExprKind::Ident(ident) => ctx.variables.get(ident.name.as_ref()).copied(),
        _ => unimplemented!("Unsupported place expression: {:?}", place_expr.pl_expr),
    }
}
