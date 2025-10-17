use melior::ir::Value;

use super::context::{create_index_constant, MlirContext};
use super::ops::build_nat_binary_operation;
use crate::ast as desc;

/// Build a natural number expression
pub fn build_nat<'ctx, 'a, 'b>(
    nat: &desc::Nat,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    use desc::Nat;
    
    match nat {
        Nat::Lit(n) => Some(create_index_constant(ctx, *n as i64)),
        Nat::Ident(ident) => ctx.variables.get(ident.name.as_ref()).copied(),
        Nat::BinOp(op, lhs, rhs) => {
            let lhs_value = build_nat(lhs, ctx)?;
            let rhs_value = build_nat(rhs, ctx)?;
            
            let result_value = build_nat_binary_operation(lhs_value, rhs_value, op, ctx);
            Some(result_value)
        }
        Nat::ThreadIdx(_) | Nat::BlockIdx(_) | Nat::BlockDim(_) 
        | Nat::WarpGrpIdx | Nat::WarpIdx | Nat::LaneIdx | Nat::GridIdx => {
            unimplemented!("GPU-specific natural numbers not yet supported in MLIR backend")
        }
        Nat::App(_, _) => {
            unimplemented!("Natural number function application not yet supported in MLIR backend")
        }
    }
}
