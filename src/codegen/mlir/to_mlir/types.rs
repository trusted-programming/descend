use crate::ast::{AtomicTy, BaseExec, DataTy, DataTyKind, FunDef, Memory, Nat, NatCtx, Ownership, ScalarTy, Ty, TyKind};
use melior::{
    dialect::func,
    ir::{
        attribute::{StringAttribute, TypeAttribute, Attribute},
        r#type::{FunctionType, IntegerType, TupleType},
        Location, Operation, Region, Type, Identifier,
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

/// Helper function to convert ScalarTy to MLIR Type
fn scalar_ty_to_mlir<'c>(scalar_ty: &ScalarTy, context: &'c Context) -> Type<'c> {
    match scalar_ty {
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
    }
}

/// Helper function to convert DataTyKind::Ident to MLIR Type
fn ident_to_mlir<'c>(ident: &crate::ast::Ident, context: &'c Context) -> Type<'c> {
    match ident.name.as_ref() {
        "i16" => IntegerType::new(context, 16).into(),
        "i8" => IntegerType::new(context, 8).into(),
        "u16" => IntegerType::new(context, 16).into(),
        _ => unimplemented!(
            "Type identifier '{}' not yet supported in MLIR conversion",
            ident.name
        ),
    }
}

/// Helper function to map BinOp to HIVM operation names
fn binop_to_hivm_operation(binop: &crate::ast::BinOp) -> &'static str {
    match binop {
        crate::ast::BinOp::Add => "hivm.hir.vadd",
        crate::ast::BinOp::Sub => unimplemented!("HIVM dialect does not have vsub operation - subtraction not supported in HIVM vector operations"),
        crate::ast::BinOp::Mul => "hivm.hir.vmul",
        crate::ast::BinOp::Div => "hivm.hir.vdiv",
        crate::ast::BinOp::Mod => "hivm.hir.vmod",
        // For now, only support arithmetic operations
        // Other operations (comparisons, logical, bitwise) can be added later
        _ => unimplemented!("HIVM operation for {:?} not yet implemented", binop),
    }
}

/// Helper function to apply HIVM address space to a memref type string
fn apply_hivm_address_space(base_str: String, mem: &Memory) -> String {
    if base_str.starts_with("memref<") {
        match mem {
            Memory::GpuGlobal | Memory::GpuShared => {
                // Use replace_range for better performance than replacen
                let mut result = base_str;
                if let Some(pos) = result.rfind('>') {
                    result.insert_str(pos, ", #hivm.address_space<gm>");
                }
                result
            }
            Memory::GpuLocal => {
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

/// Helper function to convert DataTy to element type for arrays/memrefs
fn data_ty_to_element_type<'c>(data_ty: &DataTy, context: &'c Context) -> Type<'c> {
    match &data_ty.dty {
        DataTyKind::Scalar(scalar_ty) => scalar_ty_to_mlir(scalar_ty, context),
        DataTyKind::Ident(ident) => ident_to_mlir(ident, context),
        _ => data_ty.to_mlir(context), // Fallback to full conversion for complex types
    }
}

/// Helper function to convert scalar reference to MLIR type
fn ref_scalar_to_mlir<'c>(scalar_ty: &ScalarTy, mem: &Memory, context: &'c Context) -> Type<'c> {
    // Scalar reference -> rank-0 memref
    let elem_type = scalar_ty_to_mlir(scalar_ty, context);
    let memref_str = format!("memref<{}>", elem_type);
    let base_type = Type::parse(context, &memref_str).expect("Failed to parse rank-0 memref type");

    // Add HIVM address space if needed
    let base_str = base_type.to_string();
    let final_str = apply_hivm_address_space(base_str, mem);
    parse_type_with_hivm_fallback(context, final_str, base_type)
}

/// Helper function to convert array reference to MLIR type
fn ref_array_to_mlir<'c>(
    elem_ty: &DataTy,
    size: &Nat,
    mem: &Memory,
    context: &'c Context,
) -> Type<'c> {
    // Array reference -> memref with dimensions
    let elem_type = data_ty_to_element_type(elem_ty, context);
    let dim = nat_to_dimension(size);
    let memref_str = format!("memref<{}x{}>", dim, elem_type);
    let base_type = Type::parse(context, &memref_str).expect("Failed to parse array memref type");

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
            let elem_type = scalar_ty_to_mlir(scalar_ty, context);
            let memref_str = format!("memref<{}>", elem_type);
            Type::parse(context, &memref_str).expect("Failed to parse scalar memref type")
        }
        DataTyKind::Array(elem_ty, size) | DataTyKind::ArrayShape(elem_ty, size) => {
            let elem_type = elem_ty.to_mlir(context);
            let dim = nat_to_dimension(size);
            let memref_str = format!("memref<{}x{}>", dim, elem_type);
            Type::parse(context, &memref_str).expect("Failed to parse array memref type")
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

    // Check if this is a DataTy with At type or Ref type with GPU memory
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
            DataTyKind::Scalar(scalar_ty) => scalar_ty_to_mlir(scalar_ty, context),
            DataTyKind::Atomic(atomic_ty) => match atomic_ty {
                AtomicTy::AtomicU32 => IntegerType::new(context, 32).into(),
                AtomicTy::AtomicI32 => IntegerType::new(context, 32).into(),
            },
            DataTyKind::Tuple(elem_tys) => {
                let elem_types: Vec<Type<'c>> =
                    elem_tys.iter().map(|ty| ty.to_mlir(context)).collect();
                TupleType::new(context, &elem_types).into()
            }
            DataTyKind::Ident(ident) => ident_to_mlir(ident, context),
            DataTyKind::Array(elem_ty, size) => {
                let elem_type = elem_ty.to_mlir(context);
                let dim = nat_to_dimension(size);
                let memref_str = format!("memref<{}x{}>", dim, elem_type);
                Type::parse(context, &memref_str).expect("Failed to parse memref type")
            }
            DataTyKind::ArrayShape(elem_ty, size) => {
                // ArrayShape is similar to Array but may have different semantics
                // For now, treat it the same as Array using memref
                let elem_type = elem_ty.to_mlir(context);
                let dim = nat_to_dimension(size);
                let memref_str = format!("memref<{}x{}>", dim, elem_type);
                Type::parse(context, &memref_str).expect("Failed to parse memref type")
            }
            DataTyKind::Struct(_) => {
                unimplemented!("Struct types not yet supported in MLIR conversion")
            }
            DataTyKind::At(inner, mem) => {
                // Lower inner type and, if it is a memref, attach HIVM address space for gpu.global
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
                    DataTyKind::Scalar(scalar_ty) => {
                        ref_scalar_to_mlir(scalar_ty, &ref_dty.mem, context)
                    }
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
        let attributes: Vec<(Identifier, Attribute)> = if matches!(self.exec.exec.base, BaseExec::GpuGrid(_, _)) {
            // For GPU functions, we need to create HACC attributes
            // Since we can't easily create HACC dialect attributes here, we'll use empty attributes
            // The actual GPU attributes will be handled in the string-based generation path
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

/// Parameter usage information
#[derive(Debug, Clone, PartialEq)]
struct ParameterUsage {
    pub read: bool,
    pub write: bool,
}

impl ParameterUsage {
    fn new() -> Self {
        Self { read: false, write: false }
    }
    
    fn mark_read(&mut self) {
        self.read = true;
    }
    
    fn mark_write(&mut self) {
        self.write = true;
    }
    
    fn needs_ub_allocation(&self) -> bool {
        // Only allocate ub memory if the parameter is read from
        // (parameters that are only written to can write directly to global memory)
        self.read
    }
}

/// Collect which parameters are referenced in the function body and how they are used
fn collect_parameter_usage(fun: &crate::ast::FunDef) -> std::collections::HashMap<String, ParameterUsage> {
    use crate::ast::{Expr, ExprKind, PlaceExprKind};
    use std::collections::HashMap;
    
    let mut param_usage = HashMap::new();
    
    fn walk_expr(expr: &Expr, param_usage: &mut HashMap<String, ParameterUsage>) {
        match &expr.expr {
            ExprKind::PlaceExpr(place_expr) => {
                // This is a read operation
                if let PlaceExprKind::Ident(ident) = &place_expr.pl_expr {
                    param_usage.entry(ident.name.to_string()).or_insert_with(ParameterUsage::new).mark_read();
                }
            }
            ExprKind::BinOp(_, lhs, rhs) => {
                walk_expr(lhs, param_usage);
                walk_expr(rhs, param_usage);
            }
            ExprKind::Let(_, _, value_expr) => {
                walk_expr(value_expr, param_usage);
            }
            ExprKind::Seq(exprs) => {
                for expr in exprs {
                    walk_expr(expr, param_usage);
                }
            }
            ExprKind::Assign(place_expr, value_expr) => {
                // This is a write operation
                if let PlaceExprKind::Ident(ident) = &place_expr.pl_expr {
                    param_usage.entry(ident.name.to_string()).or_insert_with(ParameterUsage::new).mark_write();
                }
                walk_expr(value_expr, param_usage);
            }
            ExprKind::IdxAssign(place_expr, _, value_expr) => {
                // This is a write operation
                if let PlaceExprKind::Ident(ident) = &place_expr.pl_expr {
                    param_usage.entry(ident.name.to_string()).or_insert_with(ParameterUsage::new).mark_write();
                }
                walk_expr(value_expr, param_usage);
            }
            ExprKind::App(_, _, args) => {
                for arg in args {
                    walk_expr(arg, param_usage);
                }
            }
            ExprKind::IfElse(cond, case_true, case_false) => {
                walk_expr(cond, param_usage);
                walk_expr(case_true, param_usage);
                walk_expr(case_false, param_usage);
            }
            ExprKind::If(cond, case_true) => {
                walk_expr(cond, param_usage);
                walk_expr(case_true, param_usage);
            }
            ExprKind::ForNat(_, _, body) => {
                walk_expr(body, param_usage);
            }
            ExprKind::Ref(_, _, place_expr) => {
                // Taking a reference is a read operation
                if let PlaceExprKind::Ident(ident) = &place_expr.pl_expr {
                    param_usage.entry(ident.name.to_string()).or_insert_with(ParameterUsage::new).mark_read();
                }
            }
            ExprKind::Unsafe(expr) => {
                walk_expr(expr, param_usage);
            }
            _ => {
                // Other expression types don't contain variable references
            }
        }
    }
    
    walk_expr(&fun.body.body, &mut param_usage);
    param_usage
}

/// Generate body operations for GPU functions
fn generate_body_operations(
    fun: &crate::ast::FunDef,
    param_to_local: &std::collections::HashMap<String, String>,
    alloc_counter: &mut usize,
    context: &Context,
) -> String {
    use crate::ast::{Expr, ExprKind, PlaceExprKind, DataTyKind, Memory, TyKind};
    
    let mut body_ops = String::new();
    
    fn walk_expr(
        expr: &Expr,
        fun: &crate::ast::FunDef,
        param_to_local: &std::collections::HashMap<String, String>,
        alloc_counter: &mut usize,
        context: &Context,
        body_ops: &mut String,
    ) -> Option<String> {
        match &expr.expr {
            ExprKind::BinOp(binop, lhs, rhs) => {
                // Process left and right operands
                let lhs_var = walk_expr(lhs, fun, param_to_local, alloc_counter, context, body_ops)?;
                let rhs_var = walk_expr(rhs, fun, param_to_local, alloc_counter, context, body_ops)?;
                
                // Generate output allocation
                let output_alloc = if *alloc_counter == 0 {
                    "%alloc".to_string()
                } else {
                    format!("%alloc_{}", *alloc_counter - 1)
                };
                *alloc_counter += 1;
                
                // Determine the type for allocation (use the type of lhs as reference)
                let lhs_type = match &lhs.expr {
                    ExprKind::PlaceExpr(place_expr) => {
                        if let PlaceExprKind::Ident(ident) = &place_expr.pl_expr {
                            // Find the parameter declaration to get its type
                            if let Some(param_decl) = fun.param_decls.iter().find(|p| p.ident.name == ident.name) {
                                if let Some(param_ty) = &param_decl.ty {
                                    // Convert the parameter type to ub address space
                                    match &param_ty.ty {
                                        TyKind::Data(data_ty) => {
                                            match &data_ty.dty {
                                                DataTyKind::At(inner, _) => {
                                                    let base_type = inner.to_mlir(context);
                                                    let base_str = base_type.to_string();
                                                    apply_hivm_address_space(base_str, &Memory::GpuLocal)
                                                }
                                                DataTyKind::Ref(ref_dty) => {
                                                    let base_type = ref_dty.dty.to_mlir(context);
                                                    let base_str = base_type.to_string();
                                                    apply_hivm_address_space(base_str, &Memory::GpuLocal)
                                                }
                                                _ => get_mlir_type_string_with_address_space(param_ty, context),
                                            }
                                        }
                                        _ => get_mlir_type_string_with_address_space(param_ty, context),
                                    }
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                };
                
                // Generate allocation
                body_ops.push_str(&format!("    {} = memref.alloc() : {}\n", output_alloc, lhs_type));
                
                // Generate HIVM operation
                let hivm_op = binop_to_hivm_operation(binop);
                body_ops.push_str(&format!(
                    "    {} ins({}, {} : {}, {}) outs({} : {})\n",
                    hivm_op,
                    lhs_var,
                    rhs_var,
                    lhs_type,
                    lhs_type,
                    output_alloc,
                    lhs_type
                ));
                
                Some(output_alloc)
            }
            ExprKind::PlaceExpr(place_expr) => {
                if let PlaceExprKind::Ident(ident) = &place_expr.pl_expr {
                    // Return the local variable name for this parameter
                    param_to_local.get(&ident.name.to_string()).cloned()
                } else {
                    None
                }
            }
            ExprKind::Assign(place_expr, value_expr) => {
                // Handle assignment: b = a
                // First, evaluate the right-hand side (value expression)
                let value_var = walk_expr(value_expr, fun, param_to_local, alloc_counter, context, body_ops)?;
                
                // Find the target parameter for the assignment
                if let PlaceExprKind::Ident(ident) = &place_expr.pl_expr {
                    // Find the parameter declaration to get its type and index
                    if let Some((param_idx, param_decl)) = fun.param_decls.iter().enumerate().find(|(_, p)| p.ident.name == ident.name) {
                        if let Some(param_ty) = &param_decl.ty {
                            // Check if we're assigning to a reference - if so, it must be unique
                            if let TyKind::Data(data_ty) = &param_ty.ty {
                                if let DataTyKind::Ref(ref_dty) = &data_ty.dty {
                                    if ref_dty.own != Ownership::Uniq {
                                        panic!(
                                            "Assignment to non-unique reference is not allowed. Expected unique reference, found {:?}",
                                            ref_dty.own
                                        );
                                    }
                                }
                            }
                            
                            // Generate the target parameter type (should be gm address space)
                            let target_type = get_mlir_type_string_with_address_space(param_ty, context);
                            
                            // Generate the source type (should be ub address space for local allocations)
                            let source_type = match &param_ty.ty {
                                TyKind::Data(data_ty) => {
                                    match &data_ty.dty {
                                        DataTyKind::At(inner, _) => {
                                            let base_type = inner.to_mlir(context);
                                            let base_str = base_type.to_string();
                                            apply_hivm_address_space(base_str, &Memory::GpuLocal)
                                        }
                                        DataTyKind::Ref(ref_dty) => {
                                            let base_type = ref_dty.dty.to_mlir(context);
                                            let base_str = base_type.to_string();
                                            apply_hivm_address_space(base_str, &Memory::GpuLocal)
                                        }
                                        _ => get_mlir_type_string_with_address_space(param_ty, context),
                                    }
                                }
                                _ => get_mlir_type_string_with_address_space(param_ty, context),
                            };
                            
                            // Generate store operation: hivm.hir.store ins(value) outs(%argN)
                            body_ops.push_str(&format!(
                                "    hivm.hir.store ins({} : {}) outs(%arg{} : {})\n",
                                value_var,
                                source_type,
                                param_idx,
                                target_type
                            ));
                        }
                    }
                }
                
                // Assignment doesn't produce a value
                None
            }
            ExprKind::Seq(exprs) => {
                // Process sequence expressions, return the result of the last expression
                let mut last_result = None;
                for expr in exprs {
                    last_result = walk_expr(expr, fun, param_to_local, alloc_counter, context, body_ops);
                }
                last_result
            }
            ExprKind::Lit(_) => {
                // Literals don't produce SSA values in this context
                None
            }
            _ => {
                // Other expression types not yet supported
                None
            }
        }
    }
    
    walk_expr(&fun.body.body, fun, param_to_local, alloc_counter, context, &mut body_ops);
    body_ops
}

/// Generate load operations for GPU parameters
/// Returns (operations_string, param_to_local_map, final_alloc_counter)
fn generate_load_operations(
    fun: &crate::ast::FunDef, 
    param_usage: &std::collections::HashMap<String, ParameterUsage>,
    context: &Context
) -> (String, std::collections::HashMap<String, String>, usize) {
    use crate::ast::{DataTyKind, Memory, TyKind};
    use std::collections::HashMap;
    
    let mut load_ops = String::new();
    let mut param_to_local = HashMap::new();
    let mut alloc_counter = 0;
    
    for (i, param) in fun.param_decls.iter().enumerate() {
        let param_name = param.ident.name.to_string();
        
        // Check if parameter is used and needs ub allocation
        let usage = match param_usage.get(&param_name) {
            Some(usage) => usage,
            None => continue, // Parameter not used at all
        };
        
        // Only allocate ub memory if the parameter is read from
        if !usage.needs_ub_allocation() {
            continue;
        }
        
        if let Some(ty) = &param.ty {
            if let TyKind::Data(data_ty) = &ty.ty {
                let needs_gpu_load = match &data_ty.dty {
                    DataTyKind::At(_, mem) => {
                        matches!(mem, Memory::GpuGlobal | Memory::GpuShared)
                    }
                    DataTyKind::Ref(ref_dty) => {
                        matches!(ref_dty.mem, Memory::GpuGlobal | Memory::GpuShared)
                    }
                    _ => false,
                };
                
                if needs_gpu_load {
                    // Generate the original type with gm address space
                    let gm_type = get_mlir_type_string_with_address_space(ty, context);
                    
                    // Generate the local type with ub address space
                    let ub_type = match &data_ty.dty {
                        DataTyKind::At(inner, _) => {
                            let base_type = inner.to_mlir(context);
                            let base_str = base_type.to_string();
                            apply_hivm_address_space(base_str, &Memory::GpuLocal)
                        }
                        DataTyKind::Ref(ref_dty) => {
                            let base_type = ref_dty.dty.to_mlir(context);
                            let base_str = base_type.to_string();
                            apply_hivm_address_space(base_str, &Memory::GpuLocal)
                        }
                        _ => gm_type.clone(),
                    };
                    
                    // Generate alloc and load operations
                    let alloc_name = if alloc_counter == 0 {
                        "%alloc".to_string()
                    } else {
                        format!("%alloc_{}", alloc_counter - 1)
                    };
                    load_ops.push_str(&format!("    {} = memref.alloc() : {}\n", alloc_name, ub_type));
                    load_ops.push_str(&format!("    hivm.hir.load ins(%arg{} : {}) outs({} : {})\n", 
                        i, gm_type, alloc_name, ub_type));
                    
                    // Map parameter to its local version
                    param_to_local.insert(param_name, alloc_name);
                    alloc_counter += 1;
                }
            }
        }
    }
    
    (load_ops, param_to_local, alloc_counter)
}

/// Generate function with body including load operations for GPU parameters
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

    // Add GPU attributes if needed
    if matches!(fun.exec.exec.base, BaseExec::GpuGrid(_, _)) {
        // TODO: When HACC dialect is registered in the MLIR context, replace this with:
        // let hacc_attributes = create_hacc_attributes(context);
        // and use the attributes with MLIR operation builders instead of string generation
        signature.push_str(" attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}");
    }

    signature.push_str(" {\n");
    
    // Collect parameter usage information
    let param_usage = collect_parameter_usage(fun);
    
    // Generate load operations for GPU parameters (only for read usage)
    let (load_ops, param_to_local, mut alloc_counter) = generate_load_operations(fun, &param_usage, context);
    signature.push_str(&load_ops);
    
    // Generate body operations (binary operations, etc.)
    let body_ops = generate_body_operations(fun, &param_to_local, &mut alloc_counter, context);
    signature.push_str(&body_ops);
    
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
                apply_hivm_address_space(base_str, mem)
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

    /// Helper function to create a minimal GPU function with GpuGrid execution context
    fn make_gpu_function() -> FunDef {
        use crate::ast::{Block, DataTy, DataTyKind, Dim, Dim1d, ExecExpr, ExecExprKind, Ident, ScalarTy, Span};
        
        FunDef {
            ident: Ident {
                name: "gpu_kernel".into(),
                span: Some(Span { begin: 0, end: 10 }),
                is_implicit: false,
            },
            generic_params: vec![],
            generic_exec: None,
            param_decls: vec![],
            ret_dty: Box::new(DataTy::new(DataTyKind::Scalar(ScalarTy::Unit))),
            exec: ExecExpr {
                exec: Box::new(ExecExprKind {
                    base: BaseExec::GpuGrid(
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
        use crate::ast::{Block, DataTy, DataTyKind, ExecExpr, ExecExprKind, Ident, ScalarTy, Span};
        
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
    fn test_gpu_function_signature_with_attributes() {
        let context = Context::new();
        let gpu_fun = make_gpu_function();
        let signature = generate_function_with_body(&gpu_fun, &context);
        
        // Check that the signature contains the GPU attributes
        assert!(signature.contains("attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}"));
        assert!(signature.contains("func.func @gpu_kernel"));
        assert!(signature.contains(") attributes"));
    }

    #[test]
    fn test_cpu_function_signature_without_attributes() {
        let context = Context::new();
        let cpu_fun = make_cpu_function();
        let signature = generate_function_with_body(&cpu_fun, &context);
        
        // Check that the signature does NOT contain GPU attributes
        assert!(!signature.contains("attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}"));
        assert!(signature.contains("func.func @cpu_function"));
        assert!(signature.contains(") {"));
        assert!(!signature.contains(") attributes"));
    }

    #[test]
    fn test_function_signature_format() {
        let context = Context::new();
        let gpu_fun = make_gpu_function();
        let signature = generate_function_with_body(&gpu_fun, &context);
        
        // Check that attributes appear in the correct position (after params, before brace)
        let lines: Vec<&str> = signature.lines().collect();
        
        // The function signature should have 3 lines: func declaration, return, closing brace
        assert_eq!(lines.len(), 3);
        
        let func_line = lines[0]; // First line should be the function declaration
        assert!(func_line.contains("func.func @gpu_kernel"));
        assert!(func_line.contains(") attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {"));
        
        // Verify the structure: function_name() attributes { ... } {
        let parts: Vec<&str> = func_line.split(") attributes").collect();
        assert_eq!(parts.len(), 2);
        assert!(parts[0].contains("@gpu_kernel"));
        assert!(parts[1].starts_with(" {hacc.entry"));
    }
}
