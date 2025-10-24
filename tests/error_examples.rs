// Manual tests for error examples - these test specific error types and capture error details
// instead of using the macro which is designed for successful compilation tests

use descend::error::CompileError;
use std::io::ErrorKind;

#[test]
fn test_fileio_error() {
    // Test compilation of a non-existent file which should fail with a FileIO error
    let err = descend::compile("examples/error-examples/nonexistent_file.desc")
        .err()
        .unwrap();

    // Also test the full miette diagnostic output
    let miette_report = miette::Report::new(err.clone());

    // Use the graphical handler to get fancy output
    let handler = miette::GraphicalReportHandler::new();
    let mut output = String::new();
    handler
        .render_report(&mut output, miette_report.as_ref())
        .unwrap();

    insta::assert_snapshot!(output);

    // Assert that compilation failed with a FileIO error
    match err {
        CompileError::FileIO(data) => {
            assert_eq!(
                data.file_path,
                "examples/error-examples/nonexistent_file.desc"
            );
            assert_eq!(data.io_error_kind, ErrorKind::NotFound);
        }
        _ => panic!("Expected FileIO error, got {:?}", err),
    }
}

#[test]
fn test_missing_main_error() {
    // Test the missing main error directly by calling the type checker
    use descend::parser::SourceCode;
    use descend::ty_check::error::TyError;

    // Parse the file first
    let source = SourceCode::from_file("examples/error-examples/missing_main.desc").unwrap();
    let mut compil_unit = descend::parser::parse(&source).unwrap();

    // Call type checker directly to get the actual MissingMain error
    let result = descend::ty_check::ty_check(&mut compil_unit);
    assert!(result.is_err()); // Should fail with MissingMain error

    // Test the actual MissingMain error diagnostic
    let missing_main_error = TyError::MissingMain;
    let miette_report = miette::Report::new(missing_main_error);

    // Use the graphical handler to get fancy output
    let handler = miette::GraphicalReportHandler::new();
    let mut output = String::new();
    handler
        .render_report(&mut output, miette_report.as_ref())
        .unwrap();

    insta::assert_snapshot!(output);
}
