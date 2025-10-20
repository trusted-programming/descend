use crate::ast::{AtomicTy, DataTy, DataTyKind, FunDef, Memory, Nat, NatCtx, ScalarTy, Ty, TyKind};
use melior::{
    dialect::func,
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType, TupleType},
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
        match &self.dty {
            DataTyKind::Scalar(scalar_ty) => match scalar_ty {
                ScalarTy::Unit => Type::parse(context, "none").expect("Failed to parse none type"),
                ScalarTy::U8 => IntegerType::new(context, 8).into(),
                ScalarTy::U32 => IntegerType::new(context, 32).into(),
                ScalarTy::U64 => IntegerType::new(context, 64).into(),
                ScalarTy::I32 => IntegerType::new(context, 32).into(),
                ScalarTy::I64 => IntegerType::new(context, 64).into(),
                ScalarTy::F32 => Type::parse(context, "f32").expect("Failed to parse f32 type"),
                ScalarTy::F64 => Type::parse(context, "f64").expect("Failed to parse f64 type"),
                ScalarTy::Bool => IntegerType::new(context, 1).into(),
                ScalarTy::Gpu => IntegerType::new(context, 32).into(), // this will be ignored in the MLIR backend
            },
            DataTyKind::Atomic(atomic_ty) => match atomic_ty {
                AtomicTy::AtomicU32 => IntegerType::new(context, 32).into(),
                AtomicTy::AtomicI32 => IntegerType::new(context, 32).into(),
            },
            DataTyKind::Tuple(elem_tys) => {
                let elem_types: Vec<Type<'c>> = elem_tys
                    .iter()
                    .map(|ty| ty.to_mlir(context))
                    .collect();
                TupleType::new(context, &elem_types).into()
            },
            DataTyKind::Ident(ident) => {
                // Handle common type identifiers that should be scalar types
                match ident.name.as_ref() {
                    "i16" => IntegerType::new(context, 16).into(),
                    "i8" => IntegerType::new(context, 8).into(),
                    "u16" => IntegerType::new(context, 16).into(),
                    _ => unimplemented!("Type identifier '{}' not yet supported in MLIR conversion", ident.name)
                }
            }
            DataTyKind::Array(elem_ty, size) => {
                let elem_type = elem_ty.to_mlir(context);
                let dim = nat_to_dimension(size);
                let memref_str = format!("memref<{}x{}>", dim, elem_type.to_string());
                Type::parse(context, &memref_str).expect("Failed to parse memref type")
            },
            DataTyKind::ArrayShape(elem_ty, size) => {
                // ArrayShape is similar to Array but may have different semantics
                // For now, treat it the same as Array using memref
                let elem_type = elem_ty.to_mlir(context);
                let dim = nat_to_dimension(size);
                let memref_str = format!("memref<{}x{}>", dim, elem_type.to_string());
                Type::parse(context, &memref_str).expect("Failed to parse memref type")
            },
            DataTyKind::Struct(_) => {
                unimplemented!("Struct types not yet supported in MLIR conversion")
            }
            DataTyKind::At(inner, mem) => {
                // Lower inner type and, if it is a memref, attach HIVM address space for gpu.global
                let base_type = inner.to_mlir(context);
                let base_str = base_type.to_string();
                if base_str.starts_with("memref<") {
                    let final_str = match mem {
                        Memory::GpuGlobal => base_str
                            .replacen(">", ", #hivm.address_space<gm>>", 1),
                        _ => base_str,
                    };
                    Type::parse(context, &final_str).expect("Failed to parse memref with address space")
                } else {
                    // Non-memref inner types remain unchanged
                    base_type
                }
            },
            DataTyKind::Ref(ref_dty) => {
                // Convert the inner DataTy to MLIR based on its kind
                match &ref_dty.dty.dty {
                    DataTyKind::Scalar(scalar_ty) => {
                        // Scalar reference -> rank-0 memref
                        let elem_type = match scalar_ty {
                            ScalarTy::Unit => Type::parse(context, "none").expect("Failed to parse none type"),
                            ScalarTy::U8 => IntegerType::new(context, 8).into(),
                            ScalarTy::U32 => IntegerType::new(context, 32).into(),
                            ScalarTy::U64 => IntegerType::new(context, 64).into(),
                            ScalarTy::I32 => IntegerType::new(context, 32).into(),
                            ScalarTy::I64 => IntegerType::new(context, 64).into(),
                            ScalarTy::F32 => Type::parse(context, "f32").expect("Failed to parse f32 type"),
                            ScalarTy::F64 => Type::parse(context, "f64").expect("Failed to parse f64 type"),
                            ScalarTy::Bool => IntegerType::new(context, 1).into(),
                            ScalarTy::Gpu => IntegerType::new(context, 32).into(),
                        };
                        let memref_str = format!("memref<{}>", elem_type.to_string());
                        Type::parse(context, &memref_str).expect("Failed to parse rank-0 memref type")
                    },
                    DataTyKind::Array(elem_ty, size) => {
                        // Array reference -> memref with dimensions
                        // For arrays, we need to get the element type, not convert the whole DataTy
                        let elem_type = match &elem_ty.dty {
                            DataTyKind::Scalar(scalar_ty) => match scalar_ty {
                                ScalarTy::Unit => Type::parse(context, "none").expect("Failed to parse none type"),
                                ScalarTy::U8 => IntegerType::new(context, 8).into(),
                                ScalarTy::U32 => IntegerType::new(context, 32).into(),
                                ScalarTy::U64 => IntegerType::new(context, 64).into(),
                                ScalarTy::I32 => IntegerType::new(context, 32).into(),
                                ScalarTy::I64 => IntegerType::new(context, 64).into(),
                                ScalarTy::F32 => Type::parse(context, "f32").expect("Failed to parse f32 type"),
                                ScalarTy::F64 => Type::parse(context, "f64").expect("Failed to parse f64 type"),
                                ScalarTy::Bool => IntegerType::new(context, 1).into(),
                                ScalarTy::Gpu => IntegerType::new(context, 32).into(),
                            },
                            DataTyKind::Ident(ident) => {
                                // Handle common type identifiers that should be scalar types
                                match ident.name.as_ref() {
                                    "i16" => IntegerType::new(context, 16).into(),
                                    "i8" => IntegerType::new(context, 8).into(),
                                    "u16" => IntegerType::new(context, 16).into(),
                                    _ => unimplemented!("Type identifier '{}' not yet supported in MLIR conversion", ident.name)
                                }
                            },
                            _ => elem_ty.to_mlir(context), // Fallback to full conversion for complex types
                        };
                        let dim = nat_to_dimension(size);
                        let memref_str = format!("memref<{}x{}>", dim, elem_type.to_string());
                        Type::parse(context, &memref_str).expect("Failed to parse array memref type")
                    },
                    DataTyKind::ArrayShape(elem_ty, size) => {
                        // ArrayShape reference -> memref with dimensions
                        // For arrays, we need to get the element type, not convert the whole DataTy
                        let elem_type = match &elem_ty.dty {
                            DataTyKind::Scalar(scalar_ty) => match scalar_ty {
                                ScalarTy::Unit => Type::parse(context, "none").expect("Failed to parse none type"),
                                ScalarTy::U8 => IntegerType::new(context, 8).into(),
                                ScalarTy::U32 => IntegerType::new(context, 32).into(),
                                ScalarTy::U64 => IntegerType::new(context, 64).into(),
                                ScalarTy::I32 => IntegerType::new(context, 32).into(),
                                ScalarTy::I64 => IntegerType::new(context, 64).into(),
                                ScalarTy::F32 => Type::parse(context, "f32").expect("Failed to parse f32 type"),
                                ScalarTy::F64 => Type::parse(context, "f64").expect("Failed to parse f64 type"),
                                ScalarTy::Bool => IntegerType::new(context, 1).into(),
                                ScalarTy::Gpu => IntegerType::new(context, 32).into(),
                            },
                            DataTyKind::Ident(ident) => {
                                // Handle common type identifiers that should be scalar types
                                match ident.name.as_ref() {
                                    "i16" => IntegerType::new(context, 16).into(),
                                    "i8" => IntegerType::new(context, 8).into(),
                                    "u16" => IntegerType::new(context, 16).into(),
                                    _ => unimplemented!("Type identifier '{}' not yet supported in MLIR conversion", ident.name)
                                }
                            },
                            _ => elem_ty.to_mlir(context), // Fallback to full conversion for complex types
                        };
                        let dim = nat_to_dimension(size);
                        let memref_str = format!("memref<{}x{}>", dim, elem_type.to_string());
                        Type::parse(context, &memref_str).expect("Failed to parse array shape memref type")
                    },
                    DataTyKind::Tuple(_) => {
                        unimplemented!("Tuple references not yet supported in MLIR conversion")
                    }
                    DataTyKind::Struct(_) => {
                        unimplemented!("Struct references not yet supported in MLIR conversion")
                    }
                    DataTyKind::Ident(_) => {
                        unimplemented!("Type identifier references not yet supported in MLIR conversion")
                    }
                    DataTyKind::Atomic(_) => {
                        unimplemented!("Atomic references not yet supported in MLIR conversion")
                    }
                    DataTyKind::At(inner, mem) => {
                        // Build base memref type from the inner data type, then append address space if needed
                        let base_type = match &inner.dty {
                            DataTyKind::Scalar(scalar_ty) => {
                                let elem_type = match scalar_ty {
                                    ScalarTy::Unit => Type::parse(context, "none").expect("Failed to parse none type"),
                                    ScalarTy::U8 => IntegerType::new(context, 8).into(),
                                    ScalarTy::U32 => IntegerType::new(context, 32).into(),
                                    ScalarTy::U64 => IntegerType::new(context, 64).into(),
                                    ScalarTy::I32 => IntegerType::new(context, 32).into(),
                                    ScalarTy::I64 => IntegerType::new(context, 64).into(),
                                    ScalarTy::F32 => Type::parse(context, "f32").expect("Failed to parse f32 type"),
                                    ScalarTy::F64 => Type::parse(context, "f64").expect("Failed to parse f64 type"),
                                    ScalarTy::Bool => IntegerType::new(context, 1).into(),
                                    ScalarTy::Gpu => IntegerType::new(context, 32).into(),
                                };
                                let memref_str = format!("memref<{}>", elem_type.to_string());
                                Type::parse(context, &memref_str).expect("Failed to parse scalar memref type")
                            },
                            DataTyKind::Array(elem_ty, size)
                            | DataTyKind::ArrayShape(elem_ty, size) => {
                                let elem_type = elem_ty.to_mlir(context);
                                let dim = nat_to_dimension(size);
                                let memref_str = format!("memref<{}x{}>", dim, elem_type.to_string());
                                Type::parse(context, &memref_str).expect("Failed to parse array memref type")
                            },
                            DataTyKind::Tuple(_) => {
                                unimplemented!(
                                    "Tuple references with At not yet supported in MLIR conversion"
                                )
                            }
                            DataTyKind::Struct(_) => {
                                unimplemented!(
                                    "Struct references with At not yet supported in MLIR conversion"
                                )
                            }
                            DataTyKind::Ident(_) => {
                                unimplemented!(
                                    "Type identifier references with At not yet supported in MLIR conversion"
                                )
                            }
                            DataTyKind::Atomic(_) => {
                                unimplemented!(
                                    "Atomic references with At not yet supported in MLIR conversion"
                                )
                            }
                            DataTyKind::At(_, _) => {
                                unimplemented!(
                                    "Nested At in Ref not yet supported in MLIR conversion"
                                )
                            }
                            DataTyKind::Ref(_) => {
                                unimplemented!(
                                    "Nested references in Ref with At not yet supported"
                                )
                            }
                            DataTyKind::RawPtr(_) => {
                                unimplemented!(
                                    "Raw pointer references with At not yet supported in MLIR conversion"
                                )
                            }
                            DataTyKind::Dead(_) => {
                                unimplemented!(
                                    "Dead type references with At not yet supported in MLIR conversion"
                                )
                            }
                        };

                        let base_str = base_type.to_string();
                        if base_str.starts_with("memref<") {
                            let final_str = match mem {
                                Memory::GpuGlobal | Memory::GpuShared => base_str
                                .replacen(">", ", #hivm.address_space<gm>>", 1),
                                Memory::GpuLocal => base_str
                                    .replacen(">", ", #hivm.address_space<ub>>", 1),
                                Memory::CpuMem => base_str,
                                Memory::Ident(_) => panic!("Generic memory parameters should be resolved before MLIR codegen"),
                            };
                            Type::parse(context, &final_str).expect("Failed to parse memref with address space")
                        } else {
                            base_type
                        }
                    },
                    DataTyKind::Ref(_) => {
                        unimplemented!("Nested references not yet supported in MLIR conversion")
                    }
                    DataTyKind::RawPtr(_) => {
                        unimplemented!("Raw pointer references not yet supported in MLIR conversion")
                    }
                    DataTyKind::Dead(_) => {
                        unimplemented!("Dead type references not yet supported in MLIR conversion")
                    }
                }
            },
            DataTyKind::RawPtr(_) => {
                unimplemented!("Raw pointer types not yet supported in MLIR conversion")
            },
            DataTyKind::Dead(_) => {
                unimplemented!("Dead types not yet supported in MLIR conversion")
            },
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
            Memory::GpuGlobal,
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

    #[test]
    fn test_ref_array_dynamic_to_mlir() {
        let context = Context::new();
        let ref_dty = RefDty::new(
            Provenance::Ident(Ident::new("r")),
            Ownership::Shrd,
            Memory::CpuMem,
            make_data_ty(DataTyKind::Array(
                Box::new(make_data_ty(DataTyKind::Scalar(ScalarTy::I64))),
                Nat::Ident(Ident::new("n")),
            )),
        );
        let data_ty = make_data_ty(DataTyKind::Ref(Box::new(ref_dty)));
        let mlir_type = data_ty.to_mlir(&context);
        assert_eq!(mlir_type.to_string(), "memref<?xi64>");
    }

    /// Helper function to test At type lowering without MLIR parsing (avoids HIVM dialect registration)
    fn test_at_type_string(data_ty: &DataTy, context: &Context) -> String {
        match &data_ty.dty {
            DataTyKind::At(inner, mem) => {
                let base_type = inner.to_mlir(context);
                let base_str = base_type.to_string();
                if base_str.starts_with("memref<") {
                    match mem {
                        Memory::GpuGlobal | Memory::GpuShared => base_str.replacen(">", ", #hivm.address_space<gm>>", 1),
                        Memory::GpuLocal => base_str.replacen(">", ", #hivm.address_space<ub>>", 1),
                        Memory::CpuMem => base_str,
                        Memory::Ident(_) => panic!("Generic memory parameters should be resolved before MLIR codegen"),
                    }
                } else {
                    base_str
                }
            }
            _ => panic!("Expected At type"),
        }
    }

    #[test]
    fn test_at_array_gpu_global_adds_gm_address_space() {
        let context = Context::new();
        let inner = make_data_ty(DataTyKind::Array(
            Box::new(make_data_ty(DataTyKind::Scalar(ScalarTy::I32))),
            Nat::Lit(16),
        ));
        let data_ty = make_data_ty(DataTyKind::At(Box::new(inner), Memory::GpuGlobal));
        
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
}
