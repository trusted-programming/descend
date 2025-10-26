// Manual tests for error examples - these test specific error types and capture error details
// instead of using the macro which is designed for successful compilation tests

use descend::error::CompileError;

#[test]
fn fileio_error() {
    // Test compilation of a non-existent file which should fail with a FileIO error
    let err = test_error_compilation("examples/error-examples/nonexistent_file.desc");
    assert!(matches!(err, CompileError::FileIO(_)));
}
#[test]
fn missing_main_error() {
    // Test compilation of a file without a main function which should fail with a MissingMain error
    let err = test_error_compilation("examples/error-examples/missing_main.desc");
    assert!(matches!(err, CompileError::MissingMain(_)));
}

#[test]
fn parse_error() {
    let err = test_error_compilation("examples/error-examples/parse_error.desc");
    assert!(matches!(err, CompileError::Parse(_)));
}

fn test_error_compilation(file_path: &str) -> CompileError {
    // Test compilation of a file which should fail with the expected error
    let result = descend::compile(file_path);
    let err: CompileError = result.err().unwrap();

    // Also test the full miette diagnostic output with source code
    let miette_report = descend::compile_with_source(file_path).err().unwrap();

    // Use the graphical handler to get fancy output
    // Disable colors in CI environment to ensure consistent snapshots
    let theme = miette::GraphicalTheme::unicode_nocolor();

    let handler = miette::GraphicalReportHandler::new_themed(theme);
    let mut output = String::new();
    handler
        .render_report(&mut output, miette_report.as_ref())
        .unwrap();

    insta::assert_snapshot!(file_path, output);
    err
}
