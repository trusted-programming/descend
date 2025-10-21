type Res = Result<(), descend::error::ErrorReported>;

use super::BACKEND;

#[test]
fn constant() -> Res {
    let output = descend::compile("examples/core/const.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn add() -> Res {
    let output = descend::compile("examples/core/add.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn lit() -> Res {
    let output = descend::compile("examples/core/lit.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn binop() -> Res {
    let output = descend::compile("examples/core/binop.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn core_unit() -> Res {
    let output = descend::compile("examples/core/unit.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn if_test() -> Res {
    let output = descend::compile("examples/core/if.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn if_else() -> Res {
    let output = descend::compile("examples/core/if_else.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn for_loop() -> Res {
    let output = descend::compile("examples/core/for_loop.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}
