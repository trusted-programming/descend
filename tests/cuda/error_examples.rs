use super::BACKEND;

#[test]
#[should_panic]
fn thread_idx_offset_error() {
    descend::compile("examples/error-examples/thread_idx_offset.desc", BACKEND).unwrap();
}
