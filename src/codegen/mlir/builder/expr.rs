use melior::ir::Value;

use super::super::error::{missing_result_error, MlirError};
use super::context::MlirContext;
use super::control_flow::{build_if, build_if_else};
use super::literal::build_literal;
use super::loops::build_for_nat;
use super::ops::build_binary_operation;
use super::place::{build_assign, build_place_expr};
use crate::ast as desc;
use crate::codegen::mlir::to_mlir::types::ToMlir;
use desc::{Expr, Ownership, Pattern, PlaceExpr, Ty};

/// Build a let binding expression
fn build_let_binding<'ctx, 'a, 'b>(
    pattern: &Pattern,
    ty: &Option<Box<Ty>>,
    value_expr: &Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Result<Option<Value<'a, 'b>>, MlirError>
where
    'ctx: 'a,
{
    // Evaluate the value expression
    let value = match build_expr(value_expr, ctx)? {
        Some(v) => v,
        None => {
            // If the value expression returns None (e.g., from Hole),
            // allocate memory based on the type annotation
            match ty {
                Some(ty) => {
                    // Convert the type to MLIR and allocate memory
                    let mlir_type = ty.to_mlir(ctx.context);
                    use super::context::alloca_memref;
                    alloca_memref(ctx, mlir_type)?
                }
                None => {
                    // No type annotation, can't allocate
                    return Ok(None);
                }
            }
        }
    };

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
    Ok(None)
}

/// Build a reference expression
fn build_ref_expr<'ctx, 'a, 'b>(
    _prv: &Option<String>,
    _own: &Ownership,
    place_expr: &PlaceExpr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Result<Option<Value<'a, 'b>>, MlirError>
where
    'ctx: 'a,
{
    // Minimal support: take address of an identifier holding an i32 value.
    // Allocate a rank-0 memref<i32>, store current value, and return the memref value.
    use super::context::{alloca_rank0_memref, memref_store_rank0};
    use desc::PlaceExprKind;

    match &place_expr.pl_expr {
        PlaceExprKind::Ident(ident) => {
            let val = ctx
                .variables
                .get(ident.name.as_ref())
                .copied()
                .ok_or_else(|| {
                    MlirError::General("address-of of unknown identifier".to_string())
                })?;
            let elem_ty = melior::ir::Type::parse(ctx.context, "i32")
                .ok_or_else(|| MlirError::TypeParseError("Failed to parse i32 type".to_string()))?;
            let mem = alloca_rank0_memref(ctx, elem_ty)?;
            memref_store_rank0(ctx, val, mem)?;
            Ok(Some(mem))
        }
        _ => unimplemented!("Reference to non-identifier places not yet supported in MLIR backend"),
    }
}

/// Build a sequence of expressions
fn build_sequence<'ctx, 'a, 'b>(
    exprs: &[Expr],
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Result<Option<Value<'a, 'b>>, MlirError>
where
    'ctx: 'a,
{
    let mut last_value = None;
    for expr in exprs {
        last_value = build_expr(expr, ctx)?;
    }
    Ok(last_value)
}

/// Build an expression using the MlirContext
pub fn build_expr<'ctx, 'a, 'b>(
    expr: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Result<Option<Value<'a, 'b>>, MlirError>
where
    'ctx: 'a,
{
    use desc::ExprKind;
    // println!("Building expression: {:?}", expr);
    match &expr.expr {
        ExprKind::Hole => Ok(None),
        ExprKind::Lit(lit) => build_literal(lit, ctx),
        ExprKind::Block(block_expr) => {
            // TODO: Process block.prvs if needed
            build_expr(&block_expr.body, ctx)
        }
        ExprKind::BinOp(op, lhs, rhs) => {
            // Build left and right operands
            let lhs_value = build_expr(lhs, ctx)?.ok_or_else(|| {
                MlirError::General("Missing left operand for binary operation".to_string())
            })?;
            let rhs_value = build_expr(rhs, ctx)?.ok_or_else(|| {
                MlirError::General("Missing right operand for binary operation".to_string())
            })?;

            // Create the appropriate arithmetic operation
            let result_value = build_binary_operation(lhs_value, rhs_value, *op, ctx);
            Ok(Some(result_value))
        }
        ExprKind::Let(pattern, ty, value_expr) => build_let_binding(pattern, ty, value_expr, ctx),
        ExprKind::PlaceExpr(place_expr) => build_place_expr(place_expr, ctx),
        ExprKind::Seq(exprs) => build_sequence(exprs, ctx),
        ExprKind::Array(_) => unimplemented!("Array expressions not yet supported in MLIR backend"),
        ExprKind::Tuple(_) => unimplemented!("Tuple expressions not yet supported in MLIR backend"),
        ExprKind::Ref(prv, own, place_expr) => build_ref_expr(prv, own, place_expr, ctx),
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
                let v = build_expr(a, ctx)?.ok_or_else(|| {
                    MlirError::General("Missing operand for function call".to_string())
                })?;
                operands.push(v);
            }

            // Determine callee result types (may be empty)
            let result_types: Vec<Type<'_>> = ctx
                .function_results
                .get(&ident.name.to_string())
                .cloned()
                .unwrap_or_else(Vec::new);

            let location = ctx.location();
            let callee = FlatSymbolRefAttribute::new(ctx.context, &ident.name);
            let call_op = func::call(ctx.context, callee, &operands, &result_types, location);
            let call_ref = ctx.current_block.append_operation(call_op);
            if result_types.is_empty() {
                Ok(None)
            } else {
                let result = call_ref
                    .result(0)
                    .map_err(|_| missing_result_error("func.call", 0))?;
                Ok(Some(result.into()))
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
        ExprKind::Unsafe(expr) => {
            // Unsafe expressions are just passthrough - recursively build the inner expression
            build_expr(expr, ctx)
        }
        ExprKind::Range(_, _) => {
            unimplemented!("Range expressions not yet supported in MLIR backend")
        }
    }
}
