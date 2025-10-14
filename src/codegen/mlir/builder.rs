use melior::{
    dialect::{arith, arith::CmpiPredicate, func},
    ir::{
        attribute::{FloatAttribute, IntegerAttribute}, 
        operation::OperationLike, 
        r#type::IntegerType, 
        Block,
        BlockLike, BlockRef, Location, Module, RegionLike, Type, Value,
    },
    Context,
};
use std::collections::HashMap;

use super::to_mlir::ToMlir;
use crate::ast as desc;
use desc::{BinOp, Pattern};


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
            let location = ctx.location();
            let result_op = match op {
                BinOp::Add => arith::addi(lhs_value, rhs_value, location),
                BinOp::Sub => arith::subi(lhs_value, rhs_value, location),
                BinOp::Mul => arith::muli(lhs_value, rhs_value, location),
                BinOp::Div => arith::divsi(lhs_value, rhs_value, location),
                BinOp::Mod => arith::remsi(lhs_value, rhs_value, location),
                BinOp::And => arith::andi(lhs_value, rhs_value, location),
                BinOp::Or => arith::ori(lhs_value, rhs_value, location),
                BinOp::Eq => arith::cmpi(ctx.context, CmpiPredicate::Eq, lhs_value, rhs_value, location),
                BinOp::Lt => arith::cmpi(ctx.context, CmpiPredicate::Slt, lhs_value, rhs_value, location),
                BinOp::Le => arith::cmpi(ctx.context, CmpiPredicate::Sle, lhs_value, rhs_value, location),
                BinOp::Gt => arith::cmpi(ctx.context, CmpiPredicate::Sgt, lhs_value, rhs_value, location),
                BinOp::Ge => arith::cmpi(ctx.context, CmpiPredicate::Sge, lhs_value, rhs_value, location),
                BinOp::Neq => arith::cmpi(ctx.context, CmpiPredicate::Ne, lhs_value, rhs_value, location),
                BinOp::Shl => arith::shli(lhs_value, rhs_value, location),
                BinOp::Shr => arith::shrsi(lhs_value, rhs_value, location),
                BinOp::BitOr => arith::ori(lhs_value, rhs_value, location),
                BinOp::BitAnd => arith::andi(lhs_value, rhs_value, location),
            };

            let op_ref = ctx.current_block.append_operation(result_op);
            Some(op_ref.result(0).unwrap().into())
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
        ExprKind::Ref(_, _, _) => unimplemented!("Reference expressions not yet supported in MLIR backend"),
        ExprKind::LetUninit(_, _, _) => unimplemented!("Uninitialized let bindings not yet supported in MLIR backend"),
        ExprKind::Assign(_, _) => unimplemented!("Assignment expressions not yet supported in MLIR backend"),
        ExprKind::IdxAssign(_, _, _) => unimplemented!("Index assignment not yet supported in MLIR backend"),
        ExprKind::App(_, _, _) => unimplemented!("Function application not yet supported in MLIR backend"),
        ExprKind::DepApp(_, _) => unimplemented!("Dependent application not yet supported in MLIR backend"),
        ExprKind::AppKernel(_) => unimplemented!("Kernel application not yet supported in MLIR backend"),
        ExprKind::IfElse(_, _, _) => unimplemented!("If-else expressions not yet supported in MLIR backend"),
        ExprKind::If(_, _) => unimplemented!("If expressions not yet supported in MLIR backend"),
        ExprKind::For(_, _, _) => unimplemented!("For loops not yet supported in MLIR backend"),
        ExprKind::ForNat(_, _, _) => unimplemented!("For-nat loops not yet supported in MLIR backend"),
        ExprKind::While(_, _) => unimplemented!("While loops not yet supported in MLIR backend"),
        ExprKind::UnOp(_, _) => unimplemented!("Unary operations not yet supported in MLIR backend"),
        ExprKind::Cast(_, _) => unimplemented!("Cast expressions not yet supported in MLIR backend"),
        ExprKind::Split(_) => unimplemented!("Split expressions not yet supported in MLIR backend"),
        ExprKind::Sched(_) => unimplemented!("Schedule expressions not yet supported in MLIR backend"),
        ExprKind::Sync(_) => unimplemented!("Sync expressions not yet supported in MLIR backend"),
        ExprKind::Unsafe(_) => unimplemented!("Unsafe expressions not yet supported in MLIR backend"),
        ExprKind::Range(_, _) => unimplemented!("Range expressions not yet supported in MLIR backend"),
    }
}

/// Helper function to create a constant operation and append it to the block
fn create_constant<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    value_attr: impl Into<melior::ir::attribute::Attribute<'ctx>>,
) -> Value<'a, 'b>
where
    'ctx: 'a,
{
    let location = ctx.location();
    let const_op = arith::constant(ctx.context, value_attr.into(), location);
    let op_ref = ctx.current_block.append_operation(const_op);
    op_ref.result(0).unwrap().into()
}

/// Helper function to create an integer constant
fn create_int_constant<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    width: u32,
    value: impl Into<i64>,
) -> Value<'a, 'b>
where
    'ctx: 'a,
{
    let int_type = IntegerType::new(ctx.context, width).into();
    let value_attr = IntegerAttribute::new(int_type, value.into());
    create_constant(ctx, value_attr)
}

/// Helper function to create a float constant
fn create_float_constant<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    type_str: &str,
    value: impl Into<f64>,
) -> Value<'a, 'b>
where
    'ctx: 'a,
{
    let float_type = Type::parse(ctx.context, type_str)
        .expect(&format!("Failed to parse {} type", type_str));
    let value_attr = FloatAttribute::new(ctx.context, float_type, value.into());
    create_constant(ctx, value_attr)
}

/// Build a literal constant
fn build_literal<'ctx, 'a, 'b>(
    lit: &desc::Lit,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    match lit {
        desc::Lit::Unit => None,
        desc::Lit::Bool(value) => Some(create_int_constant(ctx, 1, if *value { 1 } else { 0 })),
        desc::Lit::I32(value) => Some(create_int_constant(ctx, 32, *value)),
        desc::Lit::U8(value) => Some(create_int_constant(ctx, 8, *value)),
        desc::Lit::U32(value) => Some(create_int_constant(ctx, 32, *value)),
        desc::Lit::U64(value) => {
            // MLIR IntegerAttribute requires i64; reinterpret u64 bits as i64
            let int_type = IntegerType::new(ctx.context, 64).into();
            let value_attr = IntegerAttribute::new(int_type, i64::from_ne_bytes(value.to_ne_bytes()));
            Some(create_constant(ctx, value_attr))
        }
        desc::Lit::F32(value) => Some(create_float_constant(ctx, "f32", *value)),
        desc::Lit::F64(value) => Some(create_float_constant(ctx, "f64", *value)),
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
