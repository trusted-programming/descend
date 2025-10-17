use melior::{
    dialect::{arith, scf},
    ir::{
        attribute::{FloatAttribute, IntegerAttribute},
        r#type::IntegerType,
        BlockLike, BlockRef, Location, Type, Value,
    },
    Context,
};
use std::collections::HashMap;

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

/// Helper function to create a constant operation and append it to the block
pub fn create_constant<'ctx, 'a, 'b>(
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
pub fn create_int_constant<'ctx, 'a, 'b>(
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
pub fn create_float_constant<'ctx, 'a, 'b>(
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
pub fn create_index_constant<'ctx, 'a, 'b>(
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

/// Helper function to create a boolean constant
pub fn create_bool_constant<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    value: bool,
) -> Value<'a, 'b>
where
    'ctx: 'a,
{
    create_int_constant(ctx, 1, if value { 1 } else { 0 })
}

/// Helper function to append a yield operation to a block
pub fn append_yield<'a, 'b>(
    block: BlockRef<'a, 'b>,
    value: Option<Value<'a, 'b>>,
    location: Location,
) {
    let yield_op = if let Some(val) = value {
        scf::r#yield(&[val], location)
    } else {
        scf::r#yield(&[], location)
    };
    block.append_operation(yield_op);
}
