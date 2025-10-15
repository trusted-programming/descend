#![cfg(test)]

extern crate descend;

#[test]
#[should_panic]
fn thread_idx_offset_error() {
    let output = descend::compile("examples/error-examples/thread_idx_offset.desc", descend::Backend::Cuda).unwrap().0;
    insta::assert_snapshot!(output);
}