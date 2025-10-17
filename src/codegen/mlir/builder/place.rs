use melior::ir::Value;

use super::context::MlirContext;
use super::expr::build_expr;
use crate::ast as desc;

/// Build a place expression (variable lookup)
pub fn build_place_expr<'ctx, 'a, 'b>(
    place_expr: &desc::PlaceExpr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    use desc::PlaceExprKind;

    match &place_expr.pl_expr {
        PlaceExprKind::Ident(ident) => ctx.variables.get(ident.name.as_ref()).copied(),
        PlaceExprKind::View(_, _) => {
            unimplemented!("View place expressions not yet supported in MLIR backend")
        }
        PlaceExprKind::Select(_, _) => {
            unimplemented!("Select place expressions not yet supported in MLIR backend")
        }
        PlaceExprKind::Proj(_, _) => {
            unimplemented!("Projection place expressions not yet supported in MLIR backend")
        }
        PlaceExprKind::FieldProj(_, _) => {
            unimplemented!("Field projection place expressions not yet supported in MLIR backend")
        }
        PlaceExprKind::Deref(_) => {
            unimplemented!("Dereference place expressions not yet supported in MLIR backend")
        }
        PlaceExprKind::Idx(_, _) => {
            unimplemented!("Index place expressions not yet supported in MLIR backend")
        }
    }
}

/// Build an assignment expression
pub fn build_assign<'ctx, 'a, 'b>(
    place_expr: &desc::PlaceExpr,
    value_expr: &desc::Expr,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    use desc::PlaceExprKind;
    
    // Evaluate the right-hand side value
    let value = build_expr(value_expr, ctx)?;
    
    // For now, only support simple identifier assignments
    match &place_expr.pl_expr {
        PlaceExprKind::Ident(ident) => {
            // In SSA form, "assignment" is just rebinding the variable name to a new SSA value
            ctx.variables.insert(ident.name.to_string(), value);
            // Assignment expressions don't produce a value
            None
        }
        _ => {
            unimplemented!("Only simple identifier assignments are supported in MLIR backend")
        }
    }
}
