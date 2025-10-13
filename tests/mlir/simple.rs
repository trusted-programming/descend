#![cfg(test)]

extern crate descend;

type Res = Result<(), descend::error::ErrorReported>;

#[test]
fn simple() -> Res {
    let output = descend::compile("examples/simple.desc", descend::Backend::Mlir)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn add() -> Res {
    let output = descend::compile("examples/add.desc", descend::Backend::Mlir)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}
