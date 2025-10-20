use melior::{
    dialect::scf,
    ir::{
        operation::OperationBuilder, Block, BlockLike, Location, Region, RegionLike, Type, Value,
        ValueLike,
    },
};

use super::context::{create_index_constant, MlirContext};
use super::expr::build_expr;
use super::nat::build_nat;
use crate::ast as desc;

/// Build a for-nat loop
pub fn build_for_nat<'ctx, 'a, 'b>(
    ident: &desc::Ident,
    range: &desc::NatRange,
    body: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    use desc::NatRange;

    let location = ctx.location();

    // Handle only Simple range for now
    match range {
        NatRange::Simple { lower, upper } => {
            // Build lower and upper bound values
            let lower_value = build_nat(lower, ctx)?;
            let upper_value = build_nat(upper, ctx)?;

            // Create step constant (always 1 for simple range)
            let step_value = create_index_constant(ctx, 1);

            // Collect current variable values that will be loop-carried (iter_args)
            // We need to pass them as operands and receive updated values after the loop
            let parent_variables = ctx.variables.clone();
            let iter_arg_names: Vec<String> = parent_variables.keys().cloned().collect();
            let iter_arg_values: Vec<Value> = iter_arg_names
                .iter()
                .filter_map(|name| parent_variables.get(name).copied())
                .collect();
            let iter_arg_types: Vec<Type> = iter_arg_values.iter().map(|v| v.r#type()).collect();

            // Create the loop body region with block arguments:
            // - First argument: induction variable (index type)
            // - Remaining arguments: iter_args (loop-carried values)
            let index_type = Type::parse(ctx.context, "index").expect("Failed to parse index type");
            let mut block_arg_types: Vec<(Type, Location)> = vec![(index_type, location)];
            block_arg_types.extend(iter_arg_types.iter().map(|t| (*t, location)));

            let body_region = Region::new();
            let body_block = body_region.append_block(Block::new(&block_arg_types));

            // Save the current block, switch to body block
            let parent_block = ctx.current_block;
            ctx.current_block = body_block;

            // Get the induction variable (first block argument)
            let induction_var = body_block.argument(0).unwrap().into();
            ctx.variables.insert(ident.name.to_string(), induction_var);

            // Map iter_arg names to their block arguments (starting from index 1)
            for (i, name) in iter_arg_names.iter().enumerate() {
                let arg_value = body_block.argument(i + 1).unwrap().into();
                ctx.variables.insert(name.clone(), arg_value);
            }

            // Build the loop body expression
            let _body_value = build_expr(body, ctx);

            // Collect the updated values to yield
            let yield_values: Vec<Value> = iter_arg_names
                .iter()
                .filter_map(|name| ctx.variables.get(name).copied())
                .collect();

            // Add scf.yield with the updated iter_args
            let yield_op = scf::r#yield(&yield_values, location);
            body_block.append_operation(yield_op);

            // Restore the parent block
            ctx.current_block = parent_block;

            // Build the scf.for operation with iter_args
            let mut for_operands = vec![lower_value, upper_value, step_value];
            for_operands.extend(iter_arg_values);

            let for_op = OperationBuilder::new("scf.for", location)
                .add_operands(&for_operands)
                .add_results(&iter_arg_types)
                .add_regions([body_region])
                .build()
                .expect("Failed to build scf.for operation");

            // Append the for operation to the current block
            let for_op_ref = ctx.current_block.append_operation(for_op);

            // Update variables with the final values from the loop
            for (i, name) in iter_arg_names.iter().enumerate() {
                if let Ok(result) = for_op_ref.result(i) {
                    ctx.variables.insert(name.clone(), result.into());
                }
            }

            // Remove the loop induction variable
            ctx.variables.remove(&ident.name.to_string());

            // ForNat loops don't produce a value themselves
            None
        }
        NatRange::Halved { .. } => {
            unimplemented!("Halved range not yet supported in MLIR backend")
        }
        NatRange::Doubled { .. } => {
            unimplemented!("Doubled range not yet supported in MLIR backend")
        }
    }
}
