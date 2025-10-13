use crate::ast::{AtomicTy, DataTy, DataTyKind, FunDef, ScalarTy, Ty, TyKind};
use melior::{
    dialect::func,
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        r#type::FunctionType,
        Identifier, Location, Operation, Region, Type,
    },
    Context,
};

pub trait ToMlir {
    type Output<'c>;
    fn to_mlir<'c>(&self, context: &'c Context) -> Self::Output<'c>;
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
                ScalarTy::Gpu => panic!("GPU type not supported in MLIR conversion"),
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
            // Add more cases as needed
            _ => unimplemented!("Unsupported DataTy conversion: {:?}", self.dty),
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
        let function_type = FunctionType::new(context, &param_types, &[ret_ty]);

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
