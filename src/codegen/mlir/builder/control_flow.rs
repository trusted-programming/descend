use melior::{
    ir::{
        operation::OperationBuilder,
        Block, BlockLike, Region, RegionLike, Type, Value, ValueLike,
    },
};

use super::context::{MlirContext, append_yield};
use super::expr::build_expr;
use crate::ast as desc;

/// Helper function to build a branch region
pub fn build_branch_region<'ctx, 'a, 'b, F>(
    expr: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    mut build_fn: F,
) -> (Region<'ctx>, Option<Value<'a, 'b>>)
where
    'ctx: 'a,
    F: FnMut(&desc::Expr, &mut MlirContext<'ctx, 'a, 'b>) -> Option<Value<'a, 'b>>,
{
    let location = ctx.location();
    
    // Create the region with its block
    let region = Region::new();
    let block = region.append_block(Block::new(&[]));

    // Save the current block and variables, switch to new block
    let parent_block = ctx.current_block;
    let parent_variables = ctx.variables.clone();
    ctx.current_block = block;

    // Build the expression
    let result_value = build_fn(expr, ctx);

    // Add yield operation
    append_yield(block, result_value, location);

    // Restore the parent block and variables
    ctx.variables = parent_variables;
    ctx.current_block = parent_block;

    (region, result_value)
}

/// Build an if expression (without else)
pub fn build_if<'ctx, 'a, 'b>(
    cond: &desc::Expr,
    case_true: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    let location = ctx.location();

    // Build the condition value
    let cond_value = build_expr(cond, ctx)?;

    // Build the then region
    let (then_region, _true_value) = build_branch_region(case_true, ctx, build_expr);

    // Create an empty else region with its block
    let else_region = Region::new();
    let else_block = else_region.append_block(Block::new(&[]));

    // Add scf.yield to the else block
    append_yield(else_block, None, location);

    // Build the scf.if operation without result types (if without else produces no value)
    let if_op = OperationBuilder::new("scf.if", location)
        .add_operands(&[cond_value])
        .add_regions([then_region, else_region])
        .build()
        .expect("Failed to build scf.if operation");

    // Append the if operation to the current block
    ctx.current_block.append_operation(if_op);

    // If without else doesn't produce a value
    None
}

/// Build an if-else expression
pub fn build_if_else<'ctx, 'a, 'b>(
    cond: &desc::Expr,
    case_true: &desc::Expr,
    case_false: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    let location = ctx.location();

    // Build the condition value
    let cond_value = build_expr(cond, ctx)?;

    // Build the then region
    let (then_region, true_value) = build_branch_region(case_true, ctx, build_expr);

    // Build the else region
    let (else_region, false_value) = build_branch_region(case_false, ctx, build_expr);

    // Determine result types based on whether branches produce values
    let result_types: Vec<Type> = if let Some(val) = true_value {
        vec![val.r#type()]
    } else {
        vec![]
    };

    // Build the scf.if operation manually using OperationBuilder
    let if_op = OperationBuilder::new("scf.if", location)
        .add_operands(&[cond_value])
        .add_results(&result_types)
        .add_regions([then_region, else_region])
        .build()
        .expect("Failed to build scf.if operation");

    // Append the if operation to the current block
    let if_op_ref = ctx.current_block.append_operation(if_op);

    // Return the result value if the if-else produces a value
    if !result_types.is_empty() {
        Some(if_op_ref.result(0).unwrap().into())
    } else {
        None
    }
}
