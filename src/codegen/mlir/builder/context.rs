//! MLIR Context and Helper Functions
//!
//! This module provides the `MlirContext` struct for managing MLIR code generation
//! state and helper functions for creating MLIR constants and memory operations.
//!
//! # MlirContext
//!
//! The context manages:
//! - Variable bindings (name -> Value mappings)
//! - Current block for operation insertion
//! - Function result type cache for two-pass compilation
//!
//! # Helper Functions
//!
//! - `create_constant*`: Create various types of MLIR constants
//! - `alloca_memref*`: Create memory allocation operations
//! - `memref_load*`/`memref_store*`: Memory access operations
//!
//! All functions return `Result<T, MlirError>` for proper error handling.

use super::super::error::{
    missing_result_error, operation_build_error, type_parse_error, MlirError,
};
use melior::{
    dialect::{arith, scf},
    ir::{
        attribute::{FloatAttribute, IntegerAttribute},
        operation::OperationBuilder,
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
    pub function_results: HashMap<String, Vec<Type<'ctx>>>,
}

impl<'ctx, 'a, 'b> MlirContext<'ctx, 'a, 'b> {
    pub fn new(
        context: &'ctx Context,
        block: BlockRef<'a, 'b>,
        function_results: HashMap<String, Vec<Type<'ctx>>>,
    ) -> Self {
        Self {
            context,
            variables: HashMap::new(),
            current_block: block,
            function_results,
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
) -> Result<Value<'a, 'b>, MlirError>
where
    'ctx: 'a,
{
    let location = ctx.location();
    let const_op = arith::constant(ctx.context, value_attr.into(), location);
    let op_ref = ctx.current_block.append_operation(const_op);
    let result = op_ref
        .result(0)
        .map_err(|_| missing_result_error("arith.constant", 0))?;
    Ok(result.into())
}

/// Helper function to create an integer constant
pub fn create_int_constant<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    width: u32,
    value: impl Into<i64>,
) -> Result<Value<'a, 'b>, MlirError>
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
) -> Result<Value<'a, 'b>, MlirError>
where
    'ctx: 'a,
{
    let float_type =
        Type::parse(ctx.context, type_str).ok_or_else(|| type_parse_error(type_str))?;
    let value_attr = FloatAttribute::new(ctx.context, float_type, value.into());
    create_constant(ctx, value_attr)
}

/// Helper function to create an index constant
pub fn create_index_constant<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    value: impl Into<i64>,
) -> Result<Value<'a, 'b>, MlirError>
where
    'ctx: 'a,
{
    let index_type = Type::parse(ctx.context, "index").ok_or_else(|| type_parse_error("index"))?;
    let value_attr = IntegerAttribute::new(index_type, value.into());
    create_constant(ctx, value_attr)
}

/// Helper function to create a boolean constant
pub fn create_bool_constant<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    value: bool,
) -> Result<Value<'a, 'b>, MlirError>
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

/// Allocate a memref with the given type using memref.alloca
pub fn alloca_memref<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    memref_type: Type<'ctx>,
) -> Result<Value<'a, 'b>, MlirError>
where
    'ctx: 'a,
{
    let location = ctx.location();
    let op = OperationBuilder::new("memref.alloca", location)
        .add_results(&[memref_type])
        .build()
        .map_err(|_| operation_build_error("memref.alloca"))?;
    let op_ref = ctx.current_block.append_operation(op);
    let result = op_ref
        .result(0)
        .map_err(|_| missing_result_error("memref.alloca", 0))?;
    Ok(result.into())
}

/// Allocate a rank-0 memref with the given element type using memref.alloca
pub fn alloca_rank0_memref<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    elem_type: Type<'ctx>,
) -> Result<Value<'a, 'b>, MlirError>
where
    'ctx: 'a,
{
    let location = ctx.location();
    // Build memref<elem_type> type string for rank-0 memref
    let memref_ty_str = format!("memref<{}>", elem_type);
    let memref_ty =
        Type::parse(ctx.context, &memref_ty_str).ok_or_else(|| type_parse_error(&memref_ty_str))?;

    let op = OperationBuilder::new("memref.alloca", location)
        .add_results(&[memref_ty])
        .build()
        .map_err(|_| operation_build_error("memref.alloca"))?;
    let op_ref = ctx.current_block.append_operation(op);
    let result = op_ref
        .result(0)
        .map_err(|_| missing_result_error("memref.alloca", 0))?;
    Ok(result.into())
}

/// Load from a rank-0 memref using memref.load
pub fn memref_load_rank0<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    memref: Value<'a, 'b>,
    elem_type: Type<'ctx>,
) -> Result<Value<'a, 'b>, MlirError>
where
    'ctx: 'a,
{
    let location = ctx.location();
    let op = OperationBuilder::new("memref.load", location)
        .add_operands(&[memref])
        .add_results(&[elem_type])
        .build()
        .map_err(|_| operation_build_error("memref.load"))?;
    let op_ref = ctx.current_block.append_operation(op);
    let result = op_ref
        .result(0)
        .map_err(|_| missing_result_error("memref.load", 0))?;
    Ok(result.into())
}

/// Store to a rank-0 memref using memref.store
pub fn memref_store_rank0<'ctx, 'a, 'b>(
    ctx: &mut MlirContext<'ctx, 'a, 'b>,
    value: Value<'a, 'b>,
    memref: Value<'a, 'b>,
) -> Result<(), MlirError>
where
    'ctx: 'a,
{
    let location = ctx.location();
    let op = OperationBuilder::new("memref.store", location)
        .add_operands(&[value, memref])
        .build()
        .map_err(|_| operation_build_error("memref.store"))?;
    ctx.current_block.append_operation(op);
    Ok(())
}
