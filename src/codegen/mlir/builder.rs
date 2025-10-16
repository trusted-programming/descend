use melior::{
    dialect::{arith, arith::CmpiPredicate, func, scf},
    ir::{
        attribute::{FloatAttribute, IntegerAttribute},
        operation::OperationBuilder,
        operation::OperationLike,
        r#type::IntegerType,
        Block, BlockLike, BlockRef, Location, Module, Region, RegionLike, Type, Value, ValueLike,
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
                BinOp::Eq => arith::cmpi(
                    ctx.context,
                    CmpiPredicate::Eq,
                    lhs_value,
                    rhs_value,
                    location,
                ),
                BinOp::Lt => arith::cmpi(
                    ctx.context,
                    CmpiPredicate::Slt,
                    lhs_value,
                    rhs_value,
                    location,
                ),
                BinOp::Le => arith::cmpi(
                    ctx.context,
                    CmpiPredicate::Sle,
                    lhs_value,
                    rhs_value,
                    location,
                ),
                BinOp::Gt => arith::cmpi(
                    ctx.context,
                    CmpiPredicate::Sgt,
                    lhs_value,
                    rhs_value,
                    location,
                ),
                BinOp::Ge => arith::cmpi(
                    ctx.context,
                    CmpiPredicate::Sge,
                    lhs_value,
                    rhs_value,
                    location,
                ),
                BinOp::Neq => arith::cmpi(
                    ctx.context,
                    CmpiPredicate::Ne,
                    lhs_value,
                    rhs_value,
                    location,
                ),
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
        ExprKind::Ref(_, _, _) => {
            unimplemented!("Reference expressions not yet supported in MLIR backend")
        }
        ExprKind::LetUninit(_, _, _) => {
            unimplemented!("Uninitialized let bindings not yet supported in MLIR backend")
        }
        ExprKind::Assign(place_expr, value_expr) => {
            build_assign(place_expr, value_expr, ctx)
        }
        ExprKind::IdxAssign(_, _, _) => {
            unimplemented!("Index assignment not yet supported in MLIR backend")
        }
        ExprKind::App(_, _, _) => {
            unimplemented!("Function application not yet supported in MLIR backend")
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
        ExprKind::ForNat(ident, range, body) => {
            build_for_nat(ident, range, body, ctx)
        }
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
    let float_type =
        Type::parse(ctx.context, type_str).expect(&format!("Failed to parse {} type", type_str));
    let value_attr = FloatAttribute::new(ctx.context, float_type, value.into());
    create_constant(ctx, value_attr)
}

/// Helper function to create an index constant
fn create_index_constant<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    value: impl Into<i64>,
) -> Value<'a, 'b>
where
    'ctx: 'a,
{
    let index_type = Type::parse(ctx.context, "index").expect("Failed to parse index type");
    let value_attr = IntegerAttribute::new(index_type, value.into());
    create_constant(ctx, value_attr)
}

/// Build a natural number expression
fn build_nat<'ctx, 'a, 'b>(
    nat: &desc::Nat,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Option<Value<'a, 'b>>
where
    'ctx: 'a,
{
    use desc::{BinOpNat, Nat};
    
    match nat {
        Nat::Lit(n) => Some(create_index_constant(ctx, *n as i64)),
        Nat::Ident(ident) => ctx.variables.get(ident.name.as_ref()).copied(),
        Nat::BinOp(op, lhs, rhs) => {
            let lhs_value = build_nat(lhs, ctx)?;
            let rhs_value = build_nat(rhs, ctx)?;
            
            let location = ctx.location();
            let result_op = match op {
                BinOpNat::Add => arith::addi(lhs_value, rhs_value, location),
                BinOpNat::Sub => arith::subi(lhs_value, rhs_value, location),
                BinOpNat::Mul => arith::muli(lhs_value, rhs_value, location),
                BinOpNat::Div => arith::divsi(lhs_value, rhs_value, location),
                BinOpNat::Mod => arith::remsi(lhs_value, rhs_value, location),
            };
            
            let op_ref = ctx.current_block.append_operation(result_op);
            Some(op_ref.result(0).unwrap().into())
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
            let value_attr =
                IntegerAttribute::new(int_type, i64::from_ne_bytes(value.to_ne_bytes()));
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

/// Build an if expression (without else)
fn build_if<'ctx, 'a, 'b>(
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

    // Create the then region with its block
    let then_region = Region::new();
    let then_block = then_region.append_block(Block::new(&[]));

    // Save the current block and variables, switch to then block
    let parent_block = ctx.current_block;
    let parent_variables = ctx.variables.clone();
    ctx.current_block = then_block;

    // Build the true branch expression
    let _true_value = build_expr(case_true, ctx);

    // Add scf.yield to the then block (no value for if without else)
    let yield_op = scf::r#yield(&[], location);
    then_block.append_operation(yield_op);

    // Create an empty else region with its block
    let else_region = Region::new();
    let else_block = else_region.append_block(Block::new(&[]));

    // Add scf.yield to the else block
    let yield_op = scf::r#yield(&[], location);
    else_block.append_operation(yield_op);

    // Restore the parent block and variables
    ctx.variables = parent_variables;
    ctx.current_block = parent_block;

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
fn build_if_else<'ctx, 'a, 'b>(
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

    // Create the then region with its block
    let then_region = Region::new();
    let then_block = then_region.append_block(Block::new(&[]));

    // Save the current block and variables, switch to then block
    let parent_block = ctx.current_block;
    let parent_variables = ctx.variables.clone();
    ctx.current_block = then_block;

    // Build the true branch expression
    let true_value = build_expr(case_true, ctx);

    // Add scf.yield to the then block
    if let Some(val) = true_value {
        let yield_op = scf::r#yield(&[val], location);
        then_block.append_operation(yield_op);
    } else {
        let yield_op = scf::r#yield(&[], location);
        then_block.append_operation(yield_op);
    }

    // Create the else region with its block
    let else_region = Region::new();
    let else_block = else_region.append_block(Block::new(&[]));

    // Restore variables and switch to else block
    ctx.variables = parent_variables;
    ctx.current_block = else_block;

    // Build the false branch expression
    let false_value = build_expr(case_false, ctx);

    // Add scf.yield to the else block
    if let Some(val) = false_value {
        let yield_op = scf::r#yield(&[val], location);
        else_block.append_operation(yield_op);
    } else {
        let yield_op = scf::r#yield(&[], location);
        else_block.append_operation(yield_op);
    }

    // Restore the parent block
    ctx.current_block = parent_block;

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

/// Build an assignment expression
fn build_assign<'ctx, 'a, 'b>(
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

/// Build a for-nat loop
fn build_for_nat<'ctx, 'a, 'b>(
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
            let iter_arg_values: Vec<Value> = iter_arg_names.iter()
                .filter_map(|name| parent_variables.get(name).copied())
                .collect();
            let iter_arg_types: Vec<Type> = iter_arg_values.iter()
                .map(|v| v.r#type())
                .collect();
            
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
            let yield_values: Vec<Value> = iter_arg_names.iter()
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
