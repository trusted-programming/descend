type Res = Result<(), descend::error::ErrorReported>;

use super::BACKEND;

#[test]
fn simple() -> Res {
    let output = descend::compile("examples/softmax/simple.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}
