use melior::{
    dialect::{arith, arith::CmpiPredicate},
    ir::{BlockLike, Value},
};

use super::context::MlirContext;
use crate::ast as desc;
use desc::BinOp;

/// Helper function to build binary operations
pub fn build_binary_operation<'ctx, 'a, 'b>(
    lhs: Value<'a, 'b>,
    rhs: Value<'a, 'b>,
    op: BinOp,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Value<'a, 'b>
where
    'ctx: 'a,
{
    let location = ctx.location();
    let result_op = match op {
        BinOp::Add => arith::addi(lhs, rhs, location),
        BinOp::Sub => arith::subi(lhs, rhs, location),
        BinOp::Mul => arith::muli(lhs, rhs, location),
        BinOp::Div => arith::divsi(lhs, rhs, location),
        BinOp::Mod => arith::remsi(lhs, rhs, location),
        BinOp::And => arith::andi(lhs, rhs, location),
        BinOp::Or => arith::ori(lhs, rhs, location),
        BinOp::Eq => arith::cmpi(
            ctx.context,
            CmpiPredicate::Eq,
            lhs,
            rhs,
            location,
        ),
        BinOp::Lt => arith::cmpi(
            ctx.context,
            CmpiPredicate::Slt,
            lhs,
            rhs,
            location,
        ),
        BinOp::Le => arith::cmpi(
            ctx.context,
            CmpiPredicate::Sle,
            lhs,
            rhs,
            location,
        ),
        BinOp::Gt => arith::cmpi(
            ctx.context,
            CmpiPredicate::Sgt,
            lhs,
            rhs,
            location,
        ),
        BinOp::Ge => arith::cmpi(
            ctx.context,
            CmpiPredicate::Sge,
            lhs,
            rhs,
            location,
        ),
        BinOp::Neq => arith::cmpi(
            ctx.context,
            CmpiPredicate::Ne,
            lhs,
            rhs,
            location,
        ),
        BinOp::Shl => arith::shli(lhs, rhs, location),
        BinOp::Shr => arith::shrsi(lhs, rhs, location),
        BinOp::BitOr => arith::ori(lhs, rhs, location),
        BinOp::BitAnd => arith::andi(lhs, rhs, location),
    };

    let op_ref = ctx.current_block.append_operation(result_op);
    op_ref.result(0).unwrap().into()
}

/// Helper function to build binary operations for natural numbers
pub fn build_nat_binary_operation<'ctx, 'a, 'b>(
    lhs: Value<'a, 'b>,
    rhs: Value<'a, 'b>,
    op: &desc::BinOpNat,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Value<'a, 'b>
where
    'ctx: 'a,
{
    let location = ctx.location();
    let result_op = match op {
        desc::BinOpNat::Add => arith::addi(lhs, rhs, location),
        desc::BinOpNat::Sub => arith::subi(lhs, rhs, location),
        desc::BinOpNat::Mul => arith::muli(lhs, rhs, location),
        desc::BinOpNat::Div => arith::divsi(lhs, rhs, location),
        desc::BinOpNat::Mod => arith::remsi(lhs, rhs, location),
    };

    let op_ref = ctx.current_block.append_operation(result_op);
    op_ref.result(0).unwrap().into()
}
