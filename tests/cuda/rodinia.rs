type Res = Result<(), descend::error::ErrorReported>;

use super::BACKEND;

#[ignore]
#[test]
fn gaussian() -> Res {
    let output = descend::compile("examples/rodinia/gaussian.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}
