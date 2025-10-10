use melior::{
    Context,
    dialect::{DialectRegistry, arith, func, memref},
    ir::{*, attribute::{StringAttribute, TypeAttribute, IntegerAttribute}, r#type::{FunctionType, MemRefType}},
    pass::{self, ConversionPass},
    utility::{register_all_dialects, execute_passes},
    ExecutionEngine,
};

fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
}
