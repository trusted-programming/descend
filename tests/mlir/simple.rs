type Res = Result<(), descend::error::ErrorReported>;

// Automatically generate tests for all .desc files in examples/simple/
descend_derive::generate_desc_tests!();
