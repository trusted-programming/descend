use melior;

fn main() {
    melior::dialect! {
        name: "annotation",
        files: ["AscendNPU-IR/bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationBase.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationOps.td"],
        include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
    }
    
    // melior::dialect! {
    //     name: "hacc",
    //     files: ["AscendNPU-IR/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HACC/IR/HACCBase.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HACC/IR/HACCInterfaces.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td"],
    //     include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
    // }
    
    // melior::dialect! {
    //     name: "hfusion",
    //     files: ["AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionBase.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionDoc.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionTraits.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/TransformOps/HFusionTransformEnums.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.td"],
    //     include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
    // }
    
    melior::dialect! {
        name: "hivm",
        files: ["AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDoc.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/Interfaces/FlattenInterface.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.td"],
        include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
    }
    
    // melior::dialect! {
    //     name: "mathext",
    //     files: ["AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtBase.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.td"],
    //     include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
    // }
    
    // melior::dialect! {
    //     name: "memrefext",
    //     files: ["AscendNPU-IR/bishengir/include/bishengir/Dialect/MemRefExt/IR/MemRefExtBase.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/MemRefExt/IR/MemRefExtOps.td"],
    //     include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
    // }
    
    // melior::dialect! {
    //     name: "scf",
    //     files: ["AscendNPU-IR/bishengir/include/bishengir/Dialect/SCF/TransformOps/SCFTransformOps.td"],
    //     include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
    // }
    
    melior::dialect! {
        name: "symbol",
        files: ["AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolBase.td", "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.td"],
        include_directories: ["/root/descend/AscendNPU-IR/bishengir/include"],
    }
}