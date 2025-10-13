#![cfg(test)]

extern crate descend;

type Res = Result<(), descend::error::ErrorReported>;

#[ignore]
#[test]
fn gaussian() -> Res {
    let output = descend::compile("examples/rodinia/gaussian.desc", descend::Backend::Cuda)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}
