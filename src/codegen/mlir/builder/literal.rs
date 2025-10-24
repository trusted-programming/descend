use melior::ir::{Value, attribute::IntegerAttribute, r#type::IntegerType};

use super::super::error::MlirError;
use super::context::{MlirContext, create_constant, create_float_constant, create_int_constant};
use crate::ast as desc;

/// Build a literal constant
pub fn build_literal<'ctx, 'a, 'b>(
    lit: &desc::Lit,
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
) -> Result<Option<Value<'a, 'b>>, MlirError>
where
    'ctx: 'a,
{
    match lit {
        desc::Lit::Unit => Ok(None),
        desc::Lit::Bool(value) => Ok(Some(create_int_constant(
            ctx,
            1,
            if *value { 1 } else { 0 },
        )?)),
        desc::Lit::I32(value) => Ok(Some(create_int_constant(ctx, 32, *value)?)),
        desc::Lit::U8(value) => Ok(Some(create_int_constant(ctx, 8, *value)?)),
        desc::Lit::U32(value) => Ok(Some(create_int_constant(ctx, 32, *value)?)),
        desc::Lit::U64(value) => {
            // MLIR IntegerAttribute requires i64; reinterpret u64 bits as i64
            let int_type = IntegerType::new(ctx.context, 64).into();
            let value_attr =
                IntegerAttribute::new(int_type, i64::from_ne_bytes(value.to_ne_bytes()));
            Ok(Some(create_constant(ctx, value_attr)?))
        }
        desc::Lit::F32(value) => Ok(Some(create_float_constant(ctx, "f32", *value)?)),
        desc::Lit::F64(value) => Ok(Some(create_float_constant(ctx, "f64", *value)?)),
    }
}
