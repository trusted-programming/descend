# Descend

Descend is a safe systems programming language that adapts and extends Rust's type system for massively parallel computing on AI accelerators. Originally presented in the paper ["Descend: A Safe GPU Systems Programming Language"](https://arxiv.org/pdf/2305.03448), this implementation targets **Huawei Ascend NPUs** through MLIR and AscendNPU-IR as the compilation target.

Unlike unsafe languages like CUDA, OpenCL, and AscendC that rely on raw pointers and manual synchronization, Descend statically prevents data races, deadlocks, and memory safety violations through extended borrow checking, ownership tracking, and lifetime analysis.

## Key Safety Features

- **Extended Borrow Checking**: Prevents data races by tracking unique (`uniq`) and shared (`shrd`) references across thousands of parallel threads
- **Memory Views**: Safe parallel access patterns that replace raw pointer indexing, statically verified to be race-free
- **Execution Resource Tracking**: Types enforce that memory is only accessed in correct execution contexts (`cpu.thread`, `npu.grid`, `npu.block`, `npu.thread`)
- **Explicit Memory Spaces**: References track physical memory locations (`cpu.mem`, `npu.global`, `npu.shared`) preventing invalid cross-device accesses
- **Safe Synchronization**: The type system enforces correct placement and usage of synchronization primitives
- **Ownership and Lifetime Management**: Prevents use-after-free, double-free, and dangling references through compile-time ownership tracking and provenance analysis
- **Atomic Operations**: Safe concurrent operations through atomic types (`AtomicU32`, `AtomicI32`)
- **Controlled Unsafe Operations**: Unsafe blocks provide escape hatches while maintaining overall safety guarantees
- **Hierarchical Scheduling**: Safe computation scheduling over NPU execution hierarchy (grid → blocks → threads) with `sched` and `split` operations

## Target: Huawei Ascend NPU

This implementation targets **Huawei Ascend AI Processors (NPUs)** through [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR), an open-source MLIR-based intermediate representation for compiling and optimizing machine learning models on Ascend hardware.

**AscendNPU-IR provides:**

- Multi-level IR architecture for progressive lowering from high-level ML operators to hardware instructions
- MLIR-based foundation with modular and extensible compiler framework
- Ascend-specific dialects: HIVM (vector operations, DMA, synchronization), HACC (hardware optimizations), HFusion (operator fusion)

**Why MLIR for Ascend NPUs?**

- Standardized IR for integrating diverse ML frameworks with Ascend hardware
- Progressive lowering with optimization at each level
- Mature optimization infrastructure for memory layout, fusion, and performance tuning
- Cross-platform potential for other AI accelerators

## Examples

### Simple Addition

Descend:

```rust
fn main() -[t: cpu.thread]-> i32 {
    let a = 10;
    let b = 32;
    a + b
}
```

Generated MLIR:

```mlir
module {
  func.func @main() -> i32 {
    %c10_i32 = arith.constant 10 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.addi %c10_i32, %c32_i32 : i32
    return %0 : i32
  }
}
```

### NPU Memory Operations

Descend:

```rust
fn add<n: nat, r: prv>(
    a: &r shrd npu.global [i16; 16],
    b: &r shrd npu.global [i16; 16],
    c: &r uniq npu.global [i16; 16]
) -[grid: npu.grid<X<1>, X<16>>]-> () {
    // Vector addition with NPU memory spaces
    todo!()
    ()
}
```

Generated MLIR with HIVM dialect:

```mlir
module {
  func.func @add(%arg0: memref<16xi16, #hivm.address_space<gm>>, 
                 %arg1: memref<16xi16, #hivm.address_space<gm>>, 
                 %arg2: memref<16xi16, #hivm.address_space<gm>>) 
                 attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    todo!()
    return
  }
}
```

## MLIR Compiler Backend  

The MLIR backend targets Huawei Ascend NPUs through AscendNPU-IR:

- **Target**: Huawei Ascend AI processors (NPUs)
- **Output**: MLIR intermediate representation with HIVM/HACC dialects
