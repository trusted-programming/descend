#[test]
#[should_panic]
fn vec_add_memory_issue_error() {
    descend::compile("examples/error-examples/vec_add_memory_issue.desc").unwrap();
}

#[test]
#[should_panic]
fn assign_to_shared_ref_error() {
    descend::compile("examples/error-examples/assign_to_shared_ref.desc").unwrap();
}
