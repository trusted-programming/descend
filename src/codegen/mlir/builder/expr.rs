use melior::ir::Value;

use super::context::MlirContext;
use super::control_flow::{build_if, build_if_else};
use super::literal::build_literal;
use super::loops::build_for_nat;
use super::ops::build_binary_operation;
use super::place::{build_assign, build_place_expr};
use crate::ast as desc;
use desc::Pattern;

/// Build an expression using the MlirContext
pub fn build_expr<'ctx, 'a, 'b>(
    expr: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    use desc::ExprKind;
    // println!("Building expression: {:?}", expr);
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
            let result_value = build_binary_operation(lhs_value, rhs_value, *op, ctx);
            Some(result_value)
        }
        ExprKind::Let(pattern, _ty, value_expr) => {
            // Evaluate the value expression
            let value = build_expr(value_expr, ctx)?;

            // Bind the variable name to the SSA value
            match pattern {
                Pattern::Ident(_mutability, ident) => {
                    ctx.variables.insert(ident.name.to_string(), value);
                }
                Pattern::Tuple(_) => {
                    unimplemented!("Tuple patterns not yet supported in MLIR backend")
                }
                Pattern::Wildcard => {
                    // Wildcard pattern discards the value, no binding needed
                }
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
        ExprKind::Array(_) => unimplemented!("Array expressions not yet supported in MLIR backend"),
        ExprKind::Tuple(_) => unimplemented!("Tuple expressions not yet supported in MLIR backend"),
        ExprKind::Ref(_, _, _) => {
            unimplemented!("Reference expressions not yet supported in MLIR backend")
        }
        ExprKind::LetUninit(_, _, _) => {
            unimplemented!("Uninitialized let bindings not yet supported in MLIR backend")
        }
        ExprKind::Assign(place_expr, value_expr) => build_assign(place_expr, value_expr, ctx),
        ExprKind::IdxAssign(_, _, _) => {
            unimplemented!("Index assignment not yet supported in MLIR backend")
        }
        ExprKind::App(ident, _gen_args, args) => {
            use melior::{
                dialect::func,
                ir::{attribute::FlatSymbolRefAttribute, BlockLike, Type},
            };

            // Lower operands
            let mut operands: Vec<Value<'a, 'b>> = Vec::with_capacity(args.len());
            for a in args {
                let v = build_expr(a, ctx)?;
                operands.push(v);
            }

            // Determine callee result types (may be empty)
            let result_types: Vec<Type<'_>> = ctx
                .function_results
                .get(&ident.name.to_string())
                .cloned()
                .unwrap_or_else(|| Vec::new());

            let location = ctx.location();
            let callee = FlatSymbolRefAttribute::new(ctx.context, &ident.name);
            let call_op = func::call(ctx.context, callee, &operands, &result_types, location);
            let call_ref = ctx.current_block.append_operation(call_op);
            if result_types.is_empty() {
                None
            } else {
                Some(call_ref.result(0).unwrap().into())
            }
        }
        ExprKind::DepApp(_, _) => {
            unimplemented!("Dependent application not yet supported in MLIR backend")
        }
        ExprKind::AppKernel(_) => {
            unimplemented!("Kernel application not yet supported in MLIR backend")
        }
        ExprKind::IfElse(cond, case_true, case_false) => {
            build_if_else(cond, case_true, case_false, ctx)
        }
        ExprKind::If(cond, case_true) => build_if(cond, case_true, ctx),
        ExprKind::For(_, _, _) => unimplemented!("For loops not yet supported in MLIR backend"),
        ExprKind::ForNat(ident, range, body) => build_for_nat(ident, range, body, ctx),
        ExprKind::While(_, _) => unimplemented!("While loops not yet supported in MLIR backend"),
        ExprKind::UnOp(_, _) => {
            unimplemented!("Unary operations not yet supported in MLIR backend")
        }
        ExprKind::Cast(_, _) => {
            unimplemented!("Cast expressions not yet supported in MLIR backend")
        }
        ExprKind::Split(_) => unimplemented!("Split expressions not yet supported in MLIR backend"),
        ExprKind::Sched(_) => {
            unimplemented!("Schedule expressions not yet supported in MLIR backend")
        }
        ExprKind::Sync(_) => unimplemented!("Sync expressions not yet supported in MLIR backend"),
        ExprKind::Unsafe(_) => {
            unimplemented!("Unsafe expressions not yet supported in MLIR backend")
        }
        ExprKind::Range(_, _) => {
            unimplemented!("Range expressions not yet supported in MLIR backend")
        }
    }
}
