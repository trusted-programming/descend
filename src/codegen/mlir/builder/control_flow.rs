use melior::{
    dialect::scf,
    ir::{Block, BlockLike, Region, RegionLike, Type, Value, ValueLike},
};

use super::super::error::MlirError;
use super::context::{append_yield, MlirContext};
use super::expr::build_expr;
use crate::ast as desc;

/// Helper function to build a branch region
pub fn build_branch_region<'ctx, 'a, 'b, F>(
    expr: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    mut build_fn: F,
) -> Result<(Region<'ctx>, Option<Value<'a, 'b>>), MlirError>
where
    'ctx: 'a,
    F: FnMut(
        &desc::Expr,
        &mut MlirContext<'ctx, 'a, 'b>,
    ) -> Result<Option<Value<'a, 'b>>, MlirError>,
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
    let result_value = build_fn(expr, ctx)?;

    // Add yield operation
    append_yield(block, result_value, location);

    // Restore the parent block and variables
    ctx.variables = parent_variables;
    ctx.current_block = parent_block;

    Ok((region, result_value))
}

/// Build an if expression (without else)
pub fn build_if<'ctx, 'a, 'b>(
    cond: &desc::Expr,
    case_true: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Result<Option<Value<'a, 'b>>, MlirError>
where
    'ctx: 'a,
{
    let location = ctx.location();

    // Build the condition value
    let cond_value = build_expr(cond, ctx)?.ok_or_else(|| {
        MlirError::General("Missing condition value for if expression".to_string())
    })?;

    // Build the then region
    let (then_region, _true_value) = build_branch_region(case_true, ctx, build_expr)?;

    // Create an empty else region with its block
    let else_region = Region::new();
    let else_block = else_region.append_block(Block::new(&[]));

    // Add scf.yield to the else block
    append_yield(else_block, None, location);

    // Build the scf.if operation without result types
    let if_op = scf::r#if(cond_value, &[], then_region, else_region, location);
    ctx.current_block.append_operation(if_op);

    // If without else doesn't produce a value
    Ok(None)
}

/// Build an if-else expression
pub fn build_if_else<'ctx, 'a, 'b>(
    cond: &desc::Expr,
    case_true: &desc::Expr,
    case_false: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Result<Option<Value<'a, 'b>>, MlirError>
where
    'ctx: 'a,
{
    let location = ctx.location();

    // Build the condition value
    let cond_value = build_expr(cond, ctx)?.ok_or_else(|| {
        MlirError::General("Missing condition value for if-else expression".to_string())
    })?;

    // Build the then region
    let (then_region, true_value) = build_branch_region(case_true, ctx, build_expr)?;

    // Build the else region
    let (else_region, false_value) = build_branch_region(case_false, ctx, build_expr)?;

    // Determine result types based on whether branches produce values
    let result_types: Vec<Type> = if let Some(val) = true_value {
        vec![val.r#type()]
    } else {
        vec![]
    };

    // Build the scf.if operation with result types using the dialect helper
    let if_op = scf::r#if(
        cond_value,
        &result_types,
        then_region,
        else_region,
        location,
    );
    let if_op_ref = ctx.current_block.append_operation(if_op);

    // Return the result value if the if-else produces a value
    if !result_types.is_empty() {
        let result = if_op_ref
            .result(0)
            .map_err(|_| MlirError::MissingResult("if-else result missing".to_string()))?;
        Ok(Some(result.into()))
    } else {
        Ok(None)
    }
}
