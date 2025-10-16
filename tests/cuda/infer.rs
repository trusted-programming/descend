type Res = Result<(), descend::error::ErrorReported>;

use super::BACKEND;

#[test]
fn transpose() -> Res {
    let output = descend::compile("examples/infer/transpose.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn transpose_shrd_mem() -> Res {
    let output = descend::compile("examples/infer/transpose_shrd_mem.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn matmul() -> Res {
    let output = descend::compile("examples/infer/matmul.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn scale_vec() -> Res {
    let output = descend::compile("examples/infer/scale_vec.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn reverse_vec() -> Res {
    let output = descend::compile("examples/infer/reverse_vec.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn bitonic_sort() -> Res {
    let output = descend::compile("examples/infer/bitonic_sort/bitonic_sort.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn scan() -> Res {
    eprintln!(
        "Breaks because there are name clashes between nats and type variables.\n \
    This is not the case for the fully typed version.\n\
    Solution: Keep track of the kinded arguments for dependent function separately depending on their kinds."
    );
    let output = descend::compile("examples/infer/scan.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn reduce_shared_mem() -> Res {
    let output = descend::compile("examples/infer/reduce_shared_mem.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn vlc_encode() -> Res {
    let output = descend::compile("examples/infer/huffman/vlc_encode.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn vlc_encode_cg() -> Res {
    let output = descend::compile("examples/infer/huffman/vlc_encode_cg.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn vlc_encode_reuse() -> Res {
    let output = descend::compile("examples/infer/huffman/vlc_encode_reuse.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn histogram() -> Res {
    let output = descend::compile("examples/infer/huffman/histogram.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn tree_reduce() -> Res {
    let output = descend::compile("examples/infer/tree_reduce.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[test]
fn vector_add() -> Res {
    let output = descend::compile("examples/infer/vec_add.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn bfs() -> Res {
    let output = descend::compile("examples/infer/bfs.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn sgemm() -> Res {
    let output = descend::compile("examples/infer/sgemm.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}
#[ignore]
#[test]
fn shrd_mem_acc_equiv_exec() -> Res {
    let output = descend::compile("examples/shrd_mem_acc_equiv_exec.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn sssp_ffi_unsafe() -> Res {
    let output = descend::compile("examples/infer/sssp-ffi.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}

#[ignore]
#[test]
fn jacobisvd() -> Res {
    let output = descend::compile("examples/infer/jacobisvd.desc", BACKEND)?.0;
    insta::assert_snapshot!(output);
    Ok(())
}
