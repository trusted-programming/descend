#![cfg(test)]

extern crate descend;

type Res = Result<(), descend::error::ErrorReported>;

#[test]
fn transpose() -> Res {
    Ok(println!(
        "{}",
        descend::compile("examples/infer/transpose.desc", descend::Backend::Cuda)?
    ))
}

#[test]
fn transpose_shrd_mem() -> Res {
    Ok(println!(
        "{}",
        descend::compile(
            "examples/infer/transpose_shrd_mem.desc",
            descend::Backend::Cuda
        )?
    ))
}

#[test]
fn matmul() -> Res {
    Ok(println!(
        "{}",
        descend::compile("examples/infer/matmul.desc", descend::Backend::Cuda)?
    ))
}

#[test]
fn scale_vec() -> Res {
    Ok(println!(
        "{}",
        descend::compile("examples/infer/scale_vec.desc", descend::Backend::Cuda)?
    ))
}

#[test]
fn reverse_vec() -> Res {
    Ok(println!(
        "{}",
        descend::compile("examples/infer/reverse_vec.desc", descend::Backend::Cuda)?
    ))
}

#[ignore]
#[test]
fn bitonic_sort() -> Res {
    Ok(println!(
        "{}",
        descend::compile(
            "examples/infer/bitonic_sort/bitonic_sort.desc",
            descend::Backend::Cuda
        )?
    ))
}

#[test]
fn scan() -> Res {
    eprintln!(
        "Breaks because there are name clashes between nats and type variables.\n \
    This is not the case for the fully typed version.\n\
    Solution: Keep track of the kinded arguments for dependent function separately depending on their kinds."
    );
    Ok(println!(
        "{}",
        descend::compile("examples/infer/scan.desc", descend::Backend::Cuda)?
    ))
}

#[test]
fn reduce_shared_mem() -> Res {
    Ok(println!(
        "{}",
        descend::compile(
            "examples/infer/reduce_shared_mem.desc",
            descend::Backend::Cuda
        )?
    ))
}

#[test]
fn vlc_encode() -> Res {
    Ok(println!(
        "{}",
        descend::compile(
            "examples/infer/huffman/vlc_encode.desc",
            descend::Backend::Cuda
        )?
    ))
}

#[test]
fn vlc_encode_cg() -> Res {
    Ok(println!(
        "{}",
        descend::compile(
            "examples/infer/huffman/vlc_encode_cg.desc",
            descend::Backend::Cuda
        )?
    ))
}

#[test]
fn vlc_encode_reuse() -> Res {
    Ok(println!(
        "{}",
        descend::compile(
            "examples/infer/huffman/vlc_encode_reuse.desc",
            descend::Backend::Cuda
        )?
    ))
}

#[test]
fn histogram() -> Res {
    Ok(println!(
        "{}",
        descend::compile(
            "examples/infer/huffman/histogram.desc",
            descend::Backend::Cuda
        )?
    ))
}

#[test]
fn tree_reduce() -> Res {
    Ok(println!(
        "{}",
        descend::compile("examples/infer/tree_reduce.desc", descend::Backend::Cuda)?
    ))
}

#[test]
fn vector_add() -> Res {
    Ok(println!(
        "{}",
        descend::compile("examples/infer/vec_add.desc", descend::Backend::Cuda)?
    ))
}

#[ignore]
#[test]
fn bfs() -> Res {
    Ok(println!(
        "{}",
        descend::compile("examples/infer/bfs.desc", descend::Backend::Cuda)?
    ))
}

#[test]
fn sgemm() -> Res {
    Ok(println!(
        "{}",
        descend::compile("examples/infer/sgemm.desc", descend::Backend::Cuda)?
    ))
}

#[test]
fn shrd_mem_acc_equiv_exec() -> Res {
    Ok(println!(
        "{}",
        descend::compile(
            "examples/shrd_mem_acc_equiv_exec.desc",
            descend::Backend::Cuda
        )?
    ))
}

#[test]
fn sssp_ffi_unsafe() -> Res {
    Ok(println!(
        "{}",
        descend::compile("examples/infer/sssp-ffi.desc", descend::Backend::Cuda)?
    ))
}

#[test]
fn jacobisvd() -> Res {
    Ok(println!(
        "{}",
        descend::compile("examples/infer/jacobisvd.desc", descend::Backend::Cuda)?
    ))
}
