#![cfg(test)]

extern crate descend;

type Res = Result<(), descend::error::ErrorReported>;

#[test]
fn constant() -> Res {
    let output = descend::compile("examples/simple/const.desc", descend::Backend::Mlir)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn add() -> Res {
    let output = descend::compile("examples/simple/add.desc", descend::Backend::Mlir)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn lit() -> Res {
    let output = descend::compile("examples/simple/lit.desc", descend::Backend::Mlir)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn binop() -> Res {
    let output = descend::compile("examples/simple/binop.desc", descend::Backend::Mlir)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}
