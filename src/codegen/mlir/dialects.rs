// Custom MLIR Dialects
// The melior::dialect! macro generates Rust bindings for TableGen ODS files
//
// NOTE: The melior::dialect! macro requires absolute paths for include_directories
// and does NOT support env!() or concat!() macro expansion. If you encounter build errors
// about missing include files, update all include_directories paths below to match your
// project location: <YOUR_PROJECT_ROOT>/AscendNPU-IR/bishengir/include
//
// You can use the helper script: ./scripts/update-dialect-paths.sh

melior::dialect! {
    name: "annotation",
    files: [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationBase.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationOps.td",
    ],
    include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
}

melior::dialect! {
    name: "hacc",
    files: [
        "AscendNPU-IR/third-party/llvm-project/mlir/include/mlir/Dialect/DLTI/DLTIAttrs.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HACC/IR/HACCBase.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HACC/IR/HACCInterfaces.td",
    ],
    include_directories: [
        "/root/descend/AscendNPU-IR/bishengir/include",
        "/root/descend/AscendNPU-IR/third-party/llvm-project/mlir/include",
    ],
}

melior::dialect! {
    name: "hfusion",
    files: [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionBase.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionTraits.td",
    ],
    include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
}

melior::dialect! {
    name: "hivm",
    files: [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDoc.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td",
    ],
    include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
}

melior::dialect! {
    name: "mathExt",
    files: [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtBase.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.td",
    ],
    include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
}

melior::dialect! {
    name: "memref_ext",
    files: [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MemRefExt/IR/MemRefExtBase.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MemRefExt/IR/MemRefExtOps.td",
    ],
    include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
}

melior::dialect! {
    name: "symbol",
    files: [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolBase.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.td",
    ],
    include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
}
