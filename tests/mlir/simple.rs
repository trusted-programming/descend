type Res = Result<(), descend::error::ErrorReported>;

use super::BACKEND;

#[test]
fn constant() -> Res {
    let output = descend::compile("examples/simple/const.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn add() -> Res {
    let output = descend::compile("examples/simple/add.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn lit() -> Res {
    let output = descend::compile("examples/simple/lit.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn binop() -> Res {
    let output = descend::compile("examples/simple/binop.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn simple_unit() -> Res {
    let output = descend::compile("examples/simple/unit.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn if_only() -> Res {
    let output = descend::compile("examples/simple/if.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn if_else() -> Res {
    let output = descend::compile("examples/simple/if_else.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn for_loop() -> Res {
    let output = descend::compile("examples/simple/for_loop.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn func_call() -> Res {
    let output = descend::compile("examples/simple/func.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn func_params() -> Res {
    let output = descend::compile("examples/simple/func_params.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn ref_deref() -> Res {
    let output = descend::compile("examples/simple/ref_deref.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}
