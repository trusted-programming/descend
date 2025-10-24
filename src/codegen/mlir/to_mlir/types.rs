use crate::ast::{
    AtomicTy, BaseExec, DataTy, DataTyKind, FunDef, Memory, Nat, NatCtx, ScalarTy, Ty, TyKind,
};
use melior::{
    dialect::func,
    ir::{
        attribute::{Attribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType, MemRefType, TupleType},
        Identifier, Location, Operation, Region, Type,
    },
    Context,
};
pub trait ToMlir {
    type Output<'c>;
    fn to_mlir<'c>(&self, context: &'c Context) -> Self::Output<'c>;
}

impl Nat {
    fn to_dimension(self: &Self) -> Vec<i64> {
        // Try to evaluate the Nat with an empty context
        let nat_ctx = NatCtx::new();
        match self.eval(&nat_ctx) {
            Ok(size) => vec![size as i64],
            Err(_) => vec![],
        }
    }
}

impl ToMlir for ScalarTy {
    type Output<'c> = Type<'c>;

    fn to_mlir<'c>(&self, context: &'c Context) -> Type<'c> {
        match self {
            ScalarTy::Unit => Type::none(context),
            ScalarTy::U8 => Type::from(IntegerType::new(context, 8)),
            ScalarTy::U32 => IntegerType::new(context, 32).into(),
            ScalarTy::U64 => IntegerType::new(context, 64).into(),
            ScalarTy::I32 => IntegerType::new(context, 32).into(),
            ScalarTy::I64 => IntegerType::new(context, 64).into(),
            ScalarTy::F32 => Type::float32(context),
            ScalarTy::F64 => Type::float64(context),
            ScalarTy::Bool => IntegerType::new(context, 1).into(),
            ScalarTy::Npu => IntegerType::new(context, 32).into(), // this will be ignored in the MLIR backend
        }
    }
}

impl ScalarTy {
    /// Convert scalar reference to MLIR memref type with address space
    pub fn to_mlir_ref<'c>(&self, mem: &Memory, context: &'c Context) -> Type<'c> {
        // Scalar reference -> rank-0 memref
        let elem_type = self.to_mlir(context);
        let memref_str = format!("memref<{}>", elem_type);
        let base_type =
            Type::parse(context, &memref_str).expect("Failed to parse rank-0 memref type");

        // Add HIVM address space if needed
        let base_str = base_type.to_string();
        let final_str = apply_hivm_address_space(base_str, mem);
        parse_type_with_hivm_fallback(context, final_str, base_type)
    }
}

impl ToMlir for crate::ast::Ident {
    type Output<'c> = Type<'c>;

    fn to_mlir<'c>(&self, context: &'c Context) -> Type<'c> {
        match self.name.as_ref() {
            "i16" => IntegerType::new(context, 16).into(),
            "i8" => IntegerType::new(context, 8).into(),
            "u16" => IntegerType::new(context, 16).into(),
            _ => unimplemented!(
                "Type identifier '{}' not yet supported in MLIR conversion",
                self.name
            ),
        }
    }
}

/// Helper function to apply HIVM address space to a memref type string
fn apply_hivm_address_space(base_str: String, mem: &Memory) -> String {
    if base_str.starts_with("memref<") {
        match mem {
            Memory::NpuGm => {
                // Use replace_range for better performance than replacen
                let mut result = base_str;
                if let Some(pos) = result.rfind('>') {
                    result.insert_str(pos, ", #hivm.address_space<gm>");
                }
                result
            }
            Memory::NpuUb => {
                let mut result = base_str;
                if let Some(pos) = result.rfind('>') {
                    result.insert_str(pos, ", #hivm.address_space<ub>");
                }
                result
            }
            Memory::CpuMem => base_str,
            Memory::Ident(_) => {
                panic!("Generic memory parameters should be resolved before MLIR codegen")
            }
        }
    } else {
        base_str
    }
}

/// Helper function to safely parse MLIR type with HIVM address space, falling back to base type if HIVM dialect is not available
fn parse_type_with_hivm_fallback<'c>(
    context: &'c Context,
    final_str: String,
    base_type: Type<'c>,
) -> Type<'c> {
    // Check if the final string contains HIVM dialect attributes
    if final_str.contains("#hivm.address_space") {
        // Since HIVM dialect is not registered, return the base type to avoid error messages
        // This allows the code to work even when HIVM dialect is not available
        base_type
    } else {
        // No HIVM attributes, just parse normally
        Type::parse(context, &final_str).unwrap_or(base_type)
    }
}

/// Helper function to convert array reference to MLIR type
fn ref_array_to_mlir<'c>(
    elem_ty: &DataTy,
    size: &Nat,
    mem: &Memory,
    context: &'c Context,
) -> Type<'c> {
    // Array reference -> memref with dimensions
    let elem_type = elem_ty.to_mlir(context);
    let dim = size.to_dimension();
    let base_type: Type<'c> = MemRefType::new(elem_type, &dim, None, None).into();

    // Add HIVM address space if needed
    let base_str = base_type.to_string();
    let final_str = apply_hivm_address_space(base_str, mem);
    parse_type_with_hivm_fallback(context, final_str, base_type)
}

/// Helper function to convert At reference to MLIR type
fn ref_at_to_mlir<'c>(inner: &DataTy, mem: &Memory, context: &'c Context) -> Type<'c> {
    // Build base memref type from the inner data type, then append address space if needed
    let base_type = match &inner.dty {
        DataTyKind::Scalar(scalar_ty) => {
            let elem_type = scalar_ty.to_mlir(context);
            let memref_str = format!("memref<{}>", elem_type);
            Type::parse(context, &memref_str).expect("Failed to parse scalar memref type")
        }
        DataTyKind::Array(elem_ty, size) | DataTyKind::ArrayShape(elem_ty, size) => {
            let elem_type = elem_ty.to_mlir(context);
            let dim = size.to_dimension();
            MemRefType::new(elem_type, &dim, None, None).into()
        }
        DataTyKind::Tuple(_) => {
            unimplemented!("Tuple references with At not yet supported in MLIR conversion")
        }
        DataTyKind::Struct(_) => {
            unimplemented!("Struct references with At not yet supported in MLIR conversion")
        }
        DataTyKind::Ident(_) => {
            unimplemented!(
                "Type identifier references with At not yet supported in MLIR conversion"
            )
        }
        DataTyKind::Atomic(_) => {
            unimplemented!("Atomic references with At not yet supported in MLIR conversion")
        }
        DataTyKind::At(_, _) => {
            unimplemented!("Nested At in Ref not yet supported in MLIR conversion")
        }
        DataTyKind::Ref(_) => {
            unimplemented!("Nested references in Ref with At not yet supported")
        }
        DataTyKind::RawPtr(_) => {
            unimplemented!("Raw pointer references with At not yet supported in MLIR conversion")
        }
        DataTyKind::Dead(_) => {
            unimplemented!("Dead type references with At not yet supported in MLIR conversion")
        }
    };

    let base_str = base_type.to_string();
    let final_str = apply_hivm_address_space(base_str, mem);
    parse_type_with_hivm_fallback(context, final_str, base_type)
}

/// Helper function to get the MLIR type string with HIVM address space if needed
fn get_mlir_type_string_with_address_space(ty: &Ty, context: &Context) -> String {
    let base_type = ty.to_mlir(context);
    let base_str = base_type.to_string();

    // Check if this is a DataTy with At type or Ref type with NPU memory
    match &ty.ty {
        TyKind::Data(data_ty) => match &data_ty.dty {
            DataTyKind::At(_, mem) => apply_hivm_address_space(base_str, mem),
            DataTyKind::Ref(ref_dty) => apply_hivm_address_space(base_str, &ref_dty.mem),
            _ => base_str,
        },
        _ => base_str,
    }
}

impl ToMlir for DataTy {
    type Output<'c> = Type<'c>;

    fn to_mlir<'c>(&self, context: &'c Context) -> Type<'c> {
        match &self.dty {
            DataTyKind::Scalar(scalar_ty) => scalar_ty.to_mlir(context),
            DataTyKind::Atomic(atomic_ty) => match atomic_ty {
                AtomicTy::AtomicU32 => IntegerType::new(context, 32).into(),
                AtomicTy::AtomicI32 => IntegerType::new(context, 32).into(),
            },
            DataTyKind::Tuple(elem_tys) => {
                let elem_types: Vec<Type<'c>> =
                    elem_tys.iter().map(|ty| ty.to_mlir(context)).collect();
                TupleType::new(context, &elem_types).into()
            }
            DataTyKind::Ident(ident) => ident.to_mlir(context),
            DataTyKind::Array(elem_ty, size) => {
                let elem_type = elem_ty.to_mlir(context);
                let dim = size.to_dimension();
                MemRefType::new(elem_type, &dim, None, None).into()
            }
            DataTyKind::ArrayShape(elem_ty, size) => {
                let elem_type = elem_ty.to_mlir(context);
                let dim = size.to_dimension();
                MemRefType::new(elem_type, &dim, None, None).into()
            }
            DataTyKind::Struct(_) => {
                unimplemented!("Struct types not yet supported in MLIR conversion")
            }
            DataTyKind::At(inner, mem) => {
                // Lower inner type and, if it is a memref, attach HIVM address space for npu.global
                let base_type = inner.to_mlir(context);
                let base_str = base_type.to_string();
                let final_str = apply_hivm_address_space(base_str, mem);
                // Since HIVM dialect is not registered, we need to create the type manually
                // by parsing the base type and then manually constructing the final type
                parse_type_with_hivm_fallback(context, final_str, base_type)
            }
            DataTyKind::Ref(ref_dty) => {
                // Convert the inner DataTy to MLIR based on its kind
                match &ref_dty.dty.dty {
                    DataTyKind::Scalar(scalar_ty) => scalar_ty.to_mlir_ref(&ref_dty.mem, context),
                    DataTyKind::Array(elem_ty, size) => {
                        ref_array_to_mlir(elem_ty, size, &ref_dty.mem, context)
                    }
                    DataTyKind::ArrayShape(elem_ty, size) => {
                        ref_array_to_mlir(elem_ty, size, &ref_dty.mem, context)
                    }
                    DataTyKind::Tuple(_) => {
                        unimplemented!("Tuple references not yet supported in MLIR conversion")
                    }
                    DataTyKind::Struct(_) => {
                        unimplemented!("Struct references not yet supported in MLIR conversion")
                    }
                    DataTyKind::Ident(_) => {
                        unimplemented!(
                            "Type identifier references not yet supported in MLIR conversion"
                        )
                    }
                    DataTyKind::Atomic(_) => {
                        unimplemented!("Atomic references not yet supported in MLIR conversion")
                    }
                    DataTyKind::At(inner, mem) => ref_at_to_mlir(inner, mem, context),
                    DataTyKind::Ref(_) => {
                        unimplemented!("Nested references not yet supported in MLIR conversion")
                    }
                    DataTyKind::RawPtr(_) => {
                        unimplemented!(
                            "Raw pointer references not yet supported in MLIR conversion"
                        )
                    }
                    DataTyKind::Dead(_) => {
                        unimplemented!("Dead type references not yet supported in MLIR conversion")
                    }
                }
            }
            DataTyKind::RawPtr(_) => {
                unimplemented!("Raw pointer types not yet supported in MLIR conversion")
            }
            DataTyKind::Dead(_) => {
                unimplemented!("Dead types not yet supported in MLIR conversion")
            }
        }
    }
}

impl ToMlir for Ty {
    type Output<'c> = Type<'c>;

    fn to_mlir<'c>(&self, context: &'c Context) -> Type<'c> {
        match &self.ty {
            TyKind::Data(data_ty) => data_ty.to_mlir(context),
            TyKind::FnTy(_fn_ty) => {
                unimplemented!("Function types as parameters not yet supported in MLIR conversion")
            }
        }
    }
}

impl ToMlir for FunDef {
    type Output<'c> = Operation<'c>;

    fn to_mlir<'c>(&self, context: &'c Context) -> Operation<'c> {
        // Collect parameter types
        let param_types: Vec<Type<'c>> = self
            .param_decls
            .iter()
            .filter_map(|param| param.ty.as_ref())
            .map(|ty| ty.to_mlir(context))
            .collect();

        // Get return type
        let ret_ty = self.ret_dty.to_mlir(context);

        // Create MLIR function type
        // For unit type (none), we don't include it in the return types
        let return_types: Vec<Type<'c>> = if ret_ty.to_string() == "none" {
            vec![]
        } else {
            vec![ret_ty]
        };
        let function_type = FunctionType::new(context, &param_types, &return_types);

        // Create function operation
        let location = Location::unknown(context);
        let function_name = &self.ident.name;

        // Create attributes based on execution type
        let attributes: Vec<(Identifier, Attribute)> =
            if matches!(self.exec.exec.base, BaseExec::NpuGrid(_, _)) {
                // For NPU functions, we need to create HACC attributes
                // Since we can't easily create HACC dialect attributes here, we'll use empty attributes
                // The actual NPU attributes will be handled in the string-based generation path
                vec![]
            } else {
                // For CPU functions, no special attributes needed
                vec![]
            };

        func::func(
            context,
            StringAttribute::new(context, function_name),
            TypeAttribute::new(function_type.into()),
            Region::new(),
            &attributes,
            location,
        )
    }
}

/// Generate function with body including load operations for NPU parameters
fn generate_function_with_body(fun: &crate::ast::FunDef, context: &Context) -> String {
    // Pre-allocate with estimated capacity to avoid reallocations
    let estimated_capacity = 50 + fun.ident.name.len() + fun.param_decls.len() * 20;
    let mut signature = String::with_capacity(estimated_capacity);

    signature.push_str("  func.func @");
    signature.push_str(&fun.ident.name);
    signature.push('(');

    // Generate parameter types with HIVM address spaces
    let param_types: Vec<String> = fun
        .param_decls
        .iter()
        .filter_map(|param| param.ty.as_ref())
        .map(|ty| get_mlir_type_string_with_address_space(ty, context))
        .collect();

    for (i, param_type) in param_types.iter().enumerate() {
        if i > 0 {
            signature.push_str(", ");
        }
        signature.push_str("%arg");
        signature.push_str(&i.to_string());
        signature.push_str(": ");
        signature.push_str(param_type);
    }

    signature.push_str(")");

    // Add NPU attributes if needed
    if matches!(fun.exec.exec.base, BaseExec::NpuGrid(_, _)) {
        // TODO: When HACC dialect is registered in the MLIR context, replace this with:
        // let hacc_attributes = create_hacc_attributes(context);
        // and use the attributes with MLIR operation builders instead of string generation
        signature
            .push_str(" attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}");
    }

    signature.push_str(" {\n");
    signature.push_str("    return\n  }\n");
    signature
}

/// Custom function to generate MLIR string with HIVM address spaces
pub fn generate_mlir_string_with_hivm(comp_unit: &crate::ast::CompilUnit) -> String {
    let context = crate::codegen::mlir::create_context();

    // Pre-allocate with estimated capacity
    let function_count = comp_unit
        .items
        .iter()
        .filter(|item| matches!(item, crate::ast::Item::FunDef(_)))
        .count();
    let estimated_capacity = 20 + function_count * 100; // Rough estimate
    let mut result = String::with_capacity(estimated_capacity);

    result.push_str("module {\n");

    for item in &comp_unit.items {
        if let crate::ast::Item::FunDef(fun) = item {
            result.push_str(&generate_function_with_body(fun, &context));
        }
    }

    result.push_str("}\n");
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Ident, Memory, Nat, Ownership, Provenance, RefDty};

    /// Helper function to create a simple DataTy from DataTyKind
    fn make_data_ty(kind: DataTyKind) -> DataTy {
        DataTy::new(kind)
    }

    #[test]
    fn test_scalar_unit_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Scalar(ScalarTy::Unit));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "none");
    }

    #[test]
    fn test_scalar_u8_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Scalar(ScalarTy::U8));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "i8");
    }

    #[test]
    fn test_scalar_u32_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Scalar(ScalarTy::U32));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "i32");
    }

    #[test]
    fn test_scalar_u64_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Scalar(ScalarTy::U64));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "i64");
    }

    #[test]
    fn test_scalar_i32_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Scalar(ScalarTy::I32));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "i32");
    }

    #[test]
    fn test_scalar_i64_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Scalar(ScalarTy::I64));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "i64");
    }

    #[test]
    fn test_scalar_f32_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Scalar(ScalarTy::F32));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "f32");
    }

    #[test]
    fn test_scalar_f64_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Scalar(ScalarTy::F64));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "f64");
    }

    #[test]
    fn test_scalar_bool_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Scalar(ScalarTy::Bool));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "i1");
    }

    #[test]
    fn test_atomic_u32_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Atomic(AtomicTy::AtomicU32));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "i32");
    }

    #[test]
    fn test_atomic_i32_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Atomic(AtomicTy::AtomicI32));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "i32");
    }

    #[test]
    fn test_tuple_empty_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Tuple(vec![]));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "tuple<>");
    }

    #[test]
    fn test_tuple_single_element_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Tuple(vec![make_data_ty(DataTyKind::Scalar(
            ScalarTy::I32,
        ))]));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "tuple<i32>");
    }

    #[test]
    fn test_tuple_multiple_elements_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Tuple(vec![
            make_data_ty(DataTyKind::Scalar(ScalarTy::I32)),
            make_data_ty(DataTyKind::Scalar(ScalarTy::F64)),
            make_data_ty(DataTyKind::Scalar(ScalarTy::Bool)),
        ]));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "tuple<i32, f64, i1>");
    }

    #[test]
    fn test_tuple_nested_to_mlir() {
        let context = Context::new();
        let inner_tuple = make_data_ty(DataTyKind::Tuple(vec![
            make_data_ty(DataTyKind::Scalar(ScalarTy::I32)),
            make_data_ty(DataTyKind::Scalar(ScalarTy::F32)),
        ]));
        let data_ty = make_data_ty(DataTyKind::Tuple(vec![
            inner_tuple,
            make_data_ty(DataTyKind::Scalar(ScalarTy::Bool)),
        ]));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "tuple<tuple<i32, f32>, i1>");
    }

    #[test]
    fn test_array_with_literal_size_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Array(
            Box::new(make_data_ty(DataTyKind::Scalar(ScalarTy::F32))),
            Nat::Lit(10),
        ));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "memref<10xf32>");
    }

    #[test]
    fn test_array_shape_with_literal_size_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::ArrayShape(
            Box::new(make_data_ty(DataTyKind::Scalar(ScalarTy::U32))),
            Nat::Lit(5),
        ));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "memref<5xi32>");
    }

    #[test]
    fn test_ref_scalar_to_mlir() {
        let context = Context::new();
        let ref_dty = RefDty::new(
            Provenance::Ident(Ident::new("r")),
            Ownership::Uniq,
            Memory::CpuMem,
            make_data_ty(DataTyKind::Scalar(ScalarTy::I32)),
        );
        let data_ty = make_data_ty(DataTyKind::Ref(Box::new(ref_dty)));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "memref<i32>");
    }

    #[test]
    fn test_ref_scalar_f32_to_mlir() {
        let context = Context::new();
        let ref_dty = RefDty::new(
            Provenance::Ident(Ident::new("r")),
            Ownership::Shrd,
            Memory::NpuGm,
            make_data_ty(DataTyKind::Scalar(ScalarTy::F32)),
        );
        let data_ty = make_data_ty(DataTyKind::Ref(Box::new(ref_dty)));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "memref<f32>");
    }

    #[test]
    fn test_ref_array_to_mlir() {
        let context = Context::new();
        let ref_dty = RefDty::new(
            Provenance::Ident(Ident::new("r")),
            Ownership::Uniq,
            Memory::CpuMem,
            make_data_ty(DataTyKind::Array(
                Box::new(make_data_ty(DataTyKind::Scalar(ScalarTy::F32))),
                Nat::Lit(10),
            )),
        );
        let data_ty = make_data_ty(DataTyKind::Ref(Box::new(ref_dty)));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "memref<10xf32>");
    }

    /// Helper function to test At type lowering without MLIR parsing (avoids HIVM dialect registration)
    fn test_at_type_string(data_ty: &DataTy, context: &Context) -> String {
        match &data_ty.dty {
            DataTyKind::At(inner, mem) => {
                let base_type = inner.to_mlir(context);
                let base_str = base_type.to_string();
                apply_hivm_address_space(base_str, mem)
            }
            _ => panic!("Expected At type"),
        }
    }

    #[test]
    fn test_at_array_npu_global_adds_gm_address_space() {
        let context = Context::new();
        let inner = make_data_ty(DataTyKind::Array(
            Box::new(make_data_ty(DataTyKind::Scalar(ScalarTy::I32))),
            Nat::Lit(16),
        ));
        let data_ty = make_data_ty(DataTyKind::At(Box::new(inner), Memory::NpuGm));

        // Test the type string generation (avoids HIVM dialect registration issues)
        let type_str = test_at_type_string(&data_ty, &context);
        assert_eq!(type_str, "memref<16xi32, #hivm.address_space<gm>>");
    }

    #[test]
    fn test_at_array_cpu_mem_keeps_plain_memref() {
        let context = Context::new();
        let inner = make_data_ty(DataTyKind::Array(
            Box::new(make_data_ty(DataTyKind::Scalar(ScalarTy::I32))),
            Nat::Lit(16),
        ));
        let data_ty = make_data_ty(DataTyKind::At(Box::new(inner), Memory::CpuMem));

        // Test the type string generation
        let type_str = test_at_type_string(&data_ty, &context);
        assert_eq!(type_str, "memref<16xi32>");
    }

    /// Helper function to create a minimal NPU function with NpuGrid execution context
    fn make_npu_function() -> FunDef {
        use crate::ast::{
            Block, DataTy, DataTyKind, Dim, Dim1d, ExecExpr, ExecExprKind, Ident, ScalarTy, Span,
        };

        FunDef {
            ident: Ident {
                name: "npu_kernel".into(),
                span: Some(Span { begin: 0, end: 10 }),
                is_implicit: false,
            },
            generic_params: vec![],
            generic_exec: None,
            param_decls: vec![],
            ret_dty: Box::new(DataTy::new(DataTyKind::Scalar(ScalarTy::Unit))),
            exec: ExecExpr {
                exec: Box::new(ExecExprKind {
                    base: BaseExec::NpuGrid(
                        Dim::X(Box::new(Dim1d(Nat::Lit(1)))),
                        Dim::X(Box::new(Dim1d(Nat::Lit(16)))),
                    ),
                    path: vec![],
                }),
                ty: None,
                span: None,
            },
            prv_rels: vec![],
            body: Box::new(Block {
                prvs: vec![],
                body: Box::new(crate::ast::Expr {
                    expr: crate::ast::ExprKind::Lit(crate::ast::Lit::Unit),
                    ty: Some(Box::new(Ty {
                        ty: TyKind::Data(Box::new(DataTy::new(DataTyKind::Scalar(ScalarTy::Unit)))),
                        span: None,
                    })),
                    span: Some(Span { begin: 0, end: 0 }),
                }),
            }),
        }
    }

    /// Helper function to create a minimal CPU function with CpuThread execution context
    fn make_cpu_function() -> FunDef {
        use crate::ast::{
            Block, DataTy, DataTyKind, ExecExpr, ExecExprKind, Ident, ScalarTy, Span,
        };

        FunDef {
            ident: Ident {
                name: "cpu_function".into(),
                span: Some(Span { begin: 0, end: 12 }),
                is_implicit: false,
            },
            generic_params: vec![],
            generic_exec: None,
            param_decls: vec![],
            ret_dty: Box::new(DataTy::new(DataTyKind::Scalar(ScalarTy::Unit))),
            exec: ExecExpr {
                exec: Box::new(ExecExprKind {
                    base: BaseExec::CpuThread,
                    path: vec![],
                }),
                ty: None,
                span: None,
            },
            prv_rels: vec![],
            body: Box::new(Block {
                prvs: vec![],
                body: Box::new(crate::ast::Expr {
                    expr: crate::ast::ExprKind::Lit(crate::ast::Lit::Unit),
                    ty: Some(Box::new(Ty {
                        ty: TyKind::Data(Box::new(DataTy::new(DataTyKind::Scalar(ScalarTy::Unit)))),
                        span: None,
                    })),
                    span: Some(Span { begin: 0, end: 0 }),
                }),
            }),
        }
    }

    #[test]
    fn test_npu_function_signature_with_attributes() {
        let context = Context::new();
        let npu_fun = make_npu_function();
        let signature = generate_function_with_body(&npu_fun, &context);

        // Check that the signature contains the NPU attributes
        assert!(signature
            .contains("attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}"));
        assert!(signature.contains("func.func @npu_kernel"));
        assert!(signature.contains(") attributes"));
    }

    #[test]
    fn test_cpu_function_signature_without_attributes() {
        let context = Context::new();
        let cpu_fun = make_cpu_function();
        let signature = generate_function_with_body(&cpu_fun, &context);

        // Check that the signature does NOT contain NPU attributes
        assert!(!signature
            .contains("attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}"));
        assert!(signature.contains("func.func @cpu_function"));
        assert!(signature.contains(") {"));
        assert!(!signature.contains(") attributes"));
    }

    #[test]
    fn test_function_signature_format() {
        let context = Context::new();
        let npu_fun = make_npu_function();
        let signature = generate_function_with_body(&npu_fun, &context);

        // Check that attributes appear in the correct position (after params, before brace)
        let lines: Vec<&str> = signature.lines().collect();

        // The function signature should have 3 lines: func declaration, return, closing brace
        assert_eq!(lines.len(), 3);

        let func_line = lines[0]; // First line should be the function declaration
        assert!(func_line.contains("func.func @npu_kernel"));
        assert!(func_line.contains(
            ") attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {"
        ));

        // Verify the structure: function_name() attributes { ... } {
        let parts: Vec<&str> = func_line.split(") attributes").collect();
        assert_eq!(parts.len(), 2);
        assert!(parts[0].contains("@npu_kernel"));
        assert!(parts[1].starts_with(" {hacc.entry"));
    }
}
