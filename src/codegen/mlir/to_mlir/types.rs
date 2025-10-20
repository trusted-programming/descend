use crate::ast::{AtomicTy, DataTy, DataTyKind, FunDef, Nat, NatCtx, ScalarTy, Ty, TyKind};
use melior::{
    dialect::func,
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        r#type::FunctionType,
        Location, Operation, Region, Type,
    },
    Context,
};

pub trait ToMlir {
    type Output<'c>;
    fn to_mlir<'c>(&self, context: &'c Context) -> Self::Output<'c>;
}

/// Helper function to convert Nat to a dimension string for MLIR types
fn nat_to_dimension(nat: &Nat) -> String {
    // Try to evaluate the Nat with an empty context
    let nat_ctx = NatCtx::new();
    match nat.eval(&nat_ctx) {
        Ok(size) => size.to_string(),
        Err(_) => "?".to_string(), // Use dynamic dimension for non-literal Nat
    }
}

impl ToMlir for DataTy {
    type Output<'c> = Type<'c>;

    fn to_mlir<'c>(&self, context: &'c Context) -> Type<'c> {
        // Convert to type string first, then parse
        let type_str = match &self.dty {
            DataTyKind::Scalar(scalar_ty) => match scalar_ty {
                ScalarTy::Unit => "none",
                ScalarTy::U8 => "i8",
                ScalarTy::U32 => "i32",
                ScalarTy::U64 => "i64",
                ScalarTy::I32 => "i32",
                ScalarTy::I64 => "i64",
                ScalarTy::F32 => "f32",
                ScalarTy::F64 => "f64",
                ScalarTy::Bool => "i1",
                ScalarTy::Gpu => "i32", // this will be ignored in the MLIR backend
            }
            .to_string(),
            DataTyKind::Atomic(atomic_ty) => match atomic_ty {
                AtomicTy::AtomicU32 => "i32",
                AtomicTy::AtomicI32 => "i32",
            }
            .to_string(),
            DataTyKind::Tuple(elem_tys) => {
                let elem_type_strs: Vec<String> = elem_tys
                    .iter()
                    .map(|ty| ty.to_mlir(context).to_string())
                    .collect();
                format!("tuple<{}>", elem_type_strs.join(", "))
            }
            DataTyKind::Ident(_) => {
                unimplemented!("Type identifiers not yet supported in MLIR conversion")
            }
            DataTyKind::Array(elem_ty, size) => {
                let elem_type_str = elem_ty.to_mlir(context).to_string();
                let dim = nat_to_dimension(size);
                format!("memref<{}x{}>", dim, elem_type_str)
            }
            DataTyKind::ArrayShape(elem_ty, size) => {
                // ArrayShape is similar to Array but may have different semantics
                // For now, treat it the same as Array using memref
                let elem_type_str = elem_ty.to_mlir(context).to_string();
                let dim = nat_to_dimension(size);
                format!("memref<{}x{}>", dim, elem_type_str)
            }
            DataTyKind::Struct(_) => {
                unimplemented!("Struct types not yet supported in MLIR conversion")
            }
            DataTyKind::At(_, _) => {
                unimplemented!("At types (memory location) not yet supported in MLIR conversion")
            }
            DataTyKind::Ref(_) => {
                unimplemented!("Reference types not yet supported in MLIR conversion")
            }
            DataTyKind::RawPtr(_) => {
                unimplemented!("Raw pointer types not yet supported in MLIR conversion")
            }
            DataTyKind::Dead(_) => {
                unimplemented!("Dead types not yet supported in MLIR conversion")
            }
        };

        Type::parse(context, &type_str).expect(&format!("Failed to parse type: {}", type_str))
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

        func::func(
            context,
            StringAttribute::new(context, function_name),
            TypeAttribute::new(function_type.into()),
            Region::new(),
            &[],
            location,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Ident, Nat};

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
    fn test_array_with_dynamic_size_to_mlir() {
        let context = Context::new();
        let data_ty = make_data_ty(DataTyKind::Array(
            Box::new(make_data_ty(DataTyKind::Scalar(ScalarTy::I64))),
            Nat::Ident(Ident::new("n")),
        ));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "memref<?xi64>");
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
}
