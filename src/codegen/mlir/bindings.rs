use melior::{
    Context,
    dialect::DialectRegistry,
    utility::register_all_dialects,
};

fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
}
