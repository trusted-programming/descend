# Descend

Descend is a safe GPU systems programming language that adapts and extends Rust's type system for massively parallel GPU programming. Unlike unsafe languages like CUDA and OpenCL that rely on raw pointers and manual synchronization, Descend statically prevents data races, deadlocks, and memory safety violations through extended borrow checking, ownership tracking, and lifetime analysis. Originally presented in the paper ["Descend: A Safe GPU Systems Programming Language"](https://arxiv.org/pdf/2305.03448) targeting NVIDIA GPUs via CUDA, this implementation extends Descend to also support Huawei Ascend NPUs through MLIR and AscendNPU-IR.

**Key Safety Features:**

- **Extended Borrow Checking**: Prevents data races by tracking unique (`uniq`) and shared (`shrd`) references across thousands of parallel threads
- **Memory Views**: Safe parallel access patterns that replace raw pointer indexing, statically verified to be race-free
- **Execution Resource Tracking**: Types enforce that memory is only accessed in correct execution contexts (`cpu.thread`, `gpu.grid`, `gpu.block`, `gpu.thread`)
- **Explicit Memory Spaces**: References track physical memory locations (`cpu.mem`, `gpu.global`, `gpu.shared`) preventing invalid cross-device accesses
- **Safe Synchronization**: The type system enforces correct placement and usage of synchronization primitives

**Design Philosophy:**

- **Imperative Systems Programming**: Low-level control like CUDA with a safety net, not high-level functional abstractions
- **Hierarchical Scheduling**: Explicitly schedule computations over GPU's execution hierarchy (grid → blocks → threads)
- **Zero-Cost Safety**: Benchmarks show performance within 3% of hand-written CUDA while providing strong compile-time guarantees
- **Heterogeneous Computing**: Holistic programming model spanning CPU and GPU with physically separated memories reflected in the type system

## Huawei Ascend NPU Support

This implementation extends Descend to target **Huawei Ascend AI Processors (NPUs)** through [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR), an open-source MLIR-based intermediate representation developed by Huawei for compiling and optimizing machine learning models on Ascend hardware.

**What is AscendNPU-IR?**

AscendNPU-IR is Huawei's compiler infrastructure for Ascend AI processors, bridging the gap between high-level machine learning frameworks and low-level NPU instructions. It provides:

- **Multi-Level IR Architecture**: Hierarchical representation enabling progressive lowering from high-level ML operators to hardware-specific instructions
- **MLIR-Based Foundation**: Built on MLIR (Multi-Level Intermediate Representation), a modular and extensible compiler framework
- **Open-Source Compiler Infrastructure**: Publicly available toolchain for the Ascend AI ecosystem
- **Framework Integration**: Designed to work seamlessly with TensorFlow, PyTorch, and other ML frameworks

**Ascend-Specific Dialects:**

AscendNPU-IR defines custom MLIR dialects tailored to Ascend NPU capabilities:

- **HIVM** (High-Level IR for Vector Machines): 
  - Vectorized operations optimized for Ascend's architecture
  - DMA operations for efficient memory transfers
  - Synchronization primitives for parallel execution
  
- **HACC** (High-Level Accelerator Compiler):
  - Hardware-specific optimizations for Ascend NPUs
  - Core computational operations and instruction generation
  - Low-level code generation and scheduling
  
- **HFusion**: 
  - Operator fusion to reduce memory traffic
  - Performance optimization through combined operations
  
- **Supporting Dialects**: MathExt, MemRefExt, annotation, and symbol management

**Why MLIR for Ascend NPUs?**

- **Standardized IR**: Common platform for integrating diverse ML frameworks with Ascend hardware
- **Progressive Lowering**: Step-by-step transformation from high-level abstractions to hardware instructions, optimizing at each level
- **Mature Optimization Infrastructure**: Leverages MLIR's robust optimization passes for memory layout, fusion, and performance tuning
- **Cross-Platform Potential**: MLIR foundation enables potential adaptation to other AI accelerators beyond Ascend

**Descend + AscendNPU-IR Integration:**

The MLIR backend maps Descend's execution contexts (`gpu.grid`/`gpu.block`/`gpu.thread`) and memory hierarchies (`gpu.global`/`gpu.local`) to corresponding Ascend NPU constructs through AscendNPU-IR's HIVM dialect. This integration:

- Preserves Descend's compile-time safety guarantees (race freedom, memory safety, synchronization correctness)
- Generates efficient code optimized for Ascend NPU hardware
- Leverages Ascend-specific optimizations through HACC and HFusion dialects
- Enables deployment of safe, high-performance parallel programs on Huawei's AI infrastructure

## Example: Scaling a Vector

Descend:

```rust
// let reference: &'a mut i32 -- Rust reference type
// let dref: &r w m d -- Descend reference type
// r = lifetime(r) | ident  -- Lifetimes, then variable ident: prv
// w = uniq | shrd -- Uniqueness
// m = cpu.mem | gpu.global | gpu.local | ident -- Memory, then ident: mem
// d = i32 | f64 | ... | ident -- Data Type, then ident: dty
// exec = cpu.thread | gpu.grid | gpu.block | gpu.thread

fn scale_vec<n: nat>(
    h_vec: &uniq cpu.mem [f64; n]
) -[t: cpu.thread]-> () {
    let mut gpu = gpu_device(0);
    let mut a_array = gpu_alloc_copy(&uniq gpu, &shrd *h_vec);
    exec::<64, 1024>(
        &uniq gpu, // which GPU?
        (&uniq a_array,), // Input data as tuple: (&in1, &in2, ..), special case: (&in1,)
        | vec: (&uniq gpu.global [f64; n]) | -[grid: gpu.grid<X<64>, X<1024>>]-> () {
            // View -- allows shaping an array without memory access
            // can be mutable or constant, i.e., to_view_mut(&uniq ...); to_view(&shrd ...)
            // let view = to_view_mut(vec.0): [[f64; n]]
            // group_mut::<1024>(view): [[ [[f64; 1024]]; n/1024]]
            let groups = group_mut::<1024>(to_view_mut(vec.0));
            // g: &uniq gpu.global [[f64; 1024]]
            sched g in groups to block in grid {
                // v: &uniq gpu.global f64
                sched v in g to _ in block {
                    *v = *v * 3.0
                }
            }
        }
    );
    copy_to_host(&shrd a_array, h_vec)
}
```

Generated CUDA code:

```cpp
#include "descend.cuh"
/*
function declarations
*/
template <std::size_t n> auto scale_vec(descend::f64 *const h_vec) -> void;
/*
function defintions
*/
template <std::size_t n> auto scale_vec(descend::f64 *const h_vec) -> void {
  auto gpu = descend::gpu_device(0);
  auto a_array = descend::gpu_alloc_copy<descend::array<descend::f64, n>>(
      (&gpu), (&(*h_vec)));
  descend::exec<64, 1024>(
      (&gpu),
      [] __device__(descend::f64 *const p0, std::size_t n) -> void {
        {
          {
            p0[((blockIdx.x * 1024) + threadIdx.x)] =
                p0[((blockIdx.x * 1024) + threadIdx.x)] * 3.0;
          }
        }
      },
      (&a_array), n);
  descend::copy_to_host<descend::array<descend::f64, n>>((&a_array), h_vec);
}
```

## Setup

Required:

- `clang-format`: Must be found in the system path.
- `rustc` and `cargo`
- `git`
- **For MLIR backend**: MLIR installation with TableGen (included in the AscendNPU-IR submodule)

Clone the repository and compile:

```bash
git clone git@github.com:descend-lang/descend.git --recurse-submodules
cargo build
cargo test
```

### Building on Different Machines

The MLIR backend uses custom dialect definitions that require absolute paths for TableGen include directories. If you encounter build errors about missing include files when building on a different machine or in a different directory:

1. **Automatic Fix**: Run the provided script to update paths:

   ```bash
   ./scripts/update-dialect-paths.sh
   ```

2. **Manual Fix**: Edit `src/codegen/mlir/dialects.rs` and update all `include_directories` paths to:

   ```
   <YOUR_PROJECT_ROOT>/AscendNPU-IR/bishengir/include
   ```

**Why is this needed?** The `melior::dialect!` macro that generates Rust bindings from TableGen files only accepts string literals for include directories and cannot use Rust's `env!()` or `concat!()` macros for dynamic path resolution. The CI pipeline automatically runs the update script to ensure correct paths in different environments.

## Usage

The Descend compiler can be run using cargo. To see all available options:

```bash
cargo run -- -h
```

### Basic Compilation

Compile a Descend source file to CUDA (default backend):

```bash
cargo run -- path/to/your_file.desc
```

This will generate `your_file.out` in the current directory.

### Backend Selection

Descend supports multiple backends:

- **CUDA**: Generates CUDA C++ code for NVIDIA GPUs (default)
- **MLIR**: Generates MLIR IR targeting Ascend NPUs via AscendNPU-IR

Compile to a specific backend:

```bash
cargo run -- path/to/your_file.desc cuda
cargo run -- path/to/your_file.desc mlir
```

### Print AST

Print the Abstract Syntax Tree using `-p` or `--print-ast`:

```bash
cargo run -- path/to/your_file.desc -p
```

This will generate both `your_file.out` and `your_file.ast` files.

### Examples

Compile a Descend example with type inference to CUDA:

```bash
cargo run -- descend-examples/infer/scale_vec.desc
```

## Compiler Backends

Descend features a frontend-agnostic architecture that supports multiple compilation targets:

### CUDA Backend

The CUDA backend generates C++ code for NVIDIA GPUs:

- **Target**: NVIDIA CUDA-capable GPUs
- **Output**: CUDA C++ code with runtime library
- **Status**: Experimental, more complete than MLIR backend but still missing features
- **Location**: `src/codegen/cuda/`

### MLIR Backend

The MLIR backend targets Huawei Ascend NPUs through AscendNPU-IR:

- **Target**: Huawei Ascend AI processors (NPUs)
- **Output**: MLIR intermediate representation
- **Status**: Early development, actively being built
- **Location**: `src/codegen/mlir/`

#### MLIR Backend Architecture

##### AscendNPU-IR Integration

- AscendNPU-IR is Huawei's MLIR-based intermediate representation for Ascend AI processors
- Provides multi-level abstractions for compute, data movement, and synchronization
- Located at `AscendNPU-IR/` (submodule)

##### Key Components

1. **Type Conversion** (`to_mlir.rs`)
   - Descend types → MLIR types
   - Scalars → builtin types (i32, f64, i1, etc.)
   - Arrays → `memref` types
   - Tuples → MLIR tuple types

2. **MLIR Builder** (`builder.rs`)
   - Constructs MLIR IR from Descend AST
   - Function declarations and definitions
   - Expression and statement translation
   - MLIR context and module management

3. **AscendNPU-IR Dialects** (`dialects.rs`)
   - **HIVM** (Huawei Intermediate Virtual Machine): Core compute dialect
     - DMA operations for memory transfers
     - Synchronization primitives
     - Vector operations
     - Macro operations
   - **annotation**: Metadata and optimization hints
   - **symbol**: Symbol management
   - Additional dialects available: HACC, HFusion, MathExt, MemRefExt

##### Implementation Roadmap

###### Phase 1: Basic MLIR Generation (Current)

- [x] Type system conversion
- [x] Function signature generation
- [ ] Expression translation to standard MLIR dialects
- [ ] Control flow (if/else, loops)
- [ ] Memory operations (alloc, load, store)

###### Phase 2: Ascend-Specific Lowering

- [ ] Map execution contexts (`gpu.grid`/`gpu.block`/`gpu.thread`) to HIVM parallel constructs
- [ ] Translate kernel launches to HIVM task scheduling
- [ ] Map memory hierarchies (`gpu.global` → HIVM global, `gpu.local` → HIVM shared)
- [ ] Utilize HIVM DMA operations
- [ ] Add synchronization primitives

###### Phase 3: Optimization and Integration

- [ ] Hardware-specific optimizations via HACC dialect
- [ ] Operator fusion via HFusion
- [ ] Pipeline optimization and memory layout tuning
- [ ] Hardware testing and benchmarking

##### Advantages of MLIR Approach

- Leverages mature MLIR infrastructure for optimizations
- Enables gradual lowering through multiple passes
- Provides standardized interfaces for AI framework integration
- Supports cross-platform retargeting to other MLIR-supported accelerators
- Better integration with compiler toolchains

##### Testing

- Unit tests: `src/codegen/mlir/to_mlir.rs`
- Integration tests: `tests/mlir/`
- Example programs: `examples/simple/`

## Modules and Directories

### ast

- Data types and representation of the Abstract Syntax Tree: expressions and types
- Visitors for convenient tree tarversals
- Span tracks the provenance of source code in the AST

### parser

- parse a string into AST
- based on Rust PEG

### ty_check

- typing rules restrict how syntactic constructs can be used
- type inference
- borrow checking and lifetime checking are part of type checking/inference
- defines contexts that are tracked during type checking
- pre-declared function signatures for views and built-in functions, such as `exec`

### codegen

- **cuda/**: CUDA backend - data types for CUDA AST, translates Descend AST to CUDA AST, printing of the CUDA AST to C++ code
- **mlir/**: MLIR backend - type conversion to MLIR, MLIR builder, AscendNPU-IR dialect bindings for Ascend NPUs
- Supports multiple compilation targets through a unified frontend

### cuda-examples/

- Contains handwritte or generated CUDA programs
- Contains `descend.cuh`; the header file which is required in order to compile Descend programs, that were translated to CUDA, with `nvcc` (contains for example the implementation of `exec`)

### descend-examples/

- Example programs written in Descend
- Many programs exist twice, once in `with_tys` and once in `infer/`
- with_tys: programs have fully-annotated types
- infer: types in programs are mainly inferred

### examples/

- Additional example programs for testing various backends
- `simple/`: Basic examples for MLIR backend development

### AscendNPU-IR/

- Submodule containing Huawei's AscendNPU-IR MLIR dialect definitions
- HIVM, HACC, HFusion, and other Ascend-specific dialects
- Build tools and documentation for Ascend NPU compilation
- Integration tests and E2E use cases
