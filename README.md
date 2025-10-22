# Descend

Descend is a safe systems programming language that adapts and extends Rust's type system for massively parallel computing on AI accelerators. Unlike unsafe languages like CUDA and OpenCL that rely on raw pointers and manual synchronization, Descend statically prevents data races, deadlocks, and memory safety violations through extended borrow checking, ownership tracking, and lifetime analysis. Originally presented in the paper ["Descend: A Safe GPU Systems Programming Language"](https://arxiv.org/pdf/2305.03448) targeting NVIDIA GPUs via CUDA, this implementation extends Descend to target **Huawei Ascend NPUs** through MLIR and AscendNPU-IR as the primary compilation target.

**Key Safety Features:**

- **Extended Borrow Checking**: Prevents data races by tracking unique (`uniq`) and shared (`shrd`) references across thousands of parallel threads
- **Memory Views**: Safe parallel access patterns that replace raw pointer indexing, statically verified to be race-free
- **Execution Resource Tracking**: Types enforce that memory is only accessed in correct execution contexts (`cpu.thread`, `gpu.grid`, `gpu.block`, `gpu.thread`)
- **Explicit Memory Spaces**: References track physical memory locations (`cpu.mem`, `gpu.global`, `gpu.shared`) preventing invalid cross-device accesses
- **Safe Synchronization**: The type system enforces correct placement and usage of synchronization primitives

**Design Philosophy:**

- **Imperative Systems Programming**: Low-level control with a safety net, not high-level functional abstractions
- **Hierarchical Scheduling**: Explicitly schedule computations over GPU's execution hierarchy (grid → blocks → threads)
- **Zero-Cost Safety**: Compile-time guarantees without runtime overhead
- **Heterogeneous Computing**: Holistic programming model spanning CPU and NPU with physically separated memories reflected in the type system

## Primary Target: Huawei Ascend NPU

This implementation primarily targets **Huawei Ascend AI Processors (NPUs)** through [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR), an open-source MLIR-based intermediate representation developed by Huawei for compiling and optimizing machine learning models on Ascend hardware. The MLIR backend is the default and most complete compilation target.

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

## Example: Simple Addition

Descend:

```rust
fn main() -[t: cpu.thread]-> i32 {
    let a = 10;
    let b = 32;
    a + b
}
```

Generated MLIR code (default backend):

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

## Example: GPU Memory Operations

Descend:

```rust
fn add<n: nat, r: prv>(
    a: &r shrd gpu.global [i16; 16],
    b: &r shrd gpu.global [i16; 16],
    c: &r uniq gpu.global [i16; 16]
) -[grid: gpu.grid<X<1>, X<16>>]-> () {
    // Vector addition with GPU memory spaces
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
    return
  }
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

   ```text
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

- **MLIR**: Generates MLIR IR targeting Ascend NPUs via AscendNPU-IR (default, recommended)
- **CUDA**: Generates CUDA C++ code for NVIDIA GPUs (experimental, limited features)

Compile to a specific backend:

```bash
cargo run -- path/to/your_file.desc mlir    # Default MLIR backend
cargo run -- path/to/your_file.desc cuda    # Experimental CUDA backend
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

### MLIR Backend (Primary)

The MLIR backend targets Huawei Ascend NPUs through AscendNPU-IR:

- **Target**: Huawei Ascend AI processors (NPUs)
- **Output**: MLIR intermediate representation with HIVM/HACC dialects
- **Status**: ✅ **Production Ready** - Complete implementation with comprehensive testing
- **Location**: `src/codegen/mlir/`
- **Features**: Full type system, HIVM address spaces, HACC device functions, comprehensive test suite

### CUDA Backend (Experimental)

The CUDA backend generates C++ code for NVIDIA GPUs:

- **Target**: NVIDIA CUDA-capable GPUs
- **Output**: CUDA C++ code with runtime library
- **Status**: ⚠️ **Experimental** - Basic functionality, many features incomplete
- **Location**: `src/codegen/cuda/`
- **Limitations**: Limited feature support, many TODO items, not recommended for production use

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

##### Implementation Status

###### ✅ Phase 1: Basic MLIR Generation (Completed)

- [x] Type system conversion
- [x] Function signature generation
- [x] Control flow (if/else, loops)
- [x] Memory operations (alloc, load, store)
- [x] HIVM address space mapping
- [x] HACC device function attributes

###### ✅ Phase 2: Ascend-Specific Lowering (Completed)

- [x] Map execution contexts (`gpu.grid`/`gpu.block`/`gpu.thread`) to HIVM parallel constructs
- [x] Map memory hierarchies (`gpu.global` → HIVM global, `gpu.local` → HIVM shared)
- [x] HIVM dialect integration with proper address spaces
- [x] HACC entry point and device function generation
- [x] Comprehensive test suite (14 passing tests)

###### 🔄 Phase 3: Optimization and Integration (In Progress)

- [x] Basic hardware-specific optimizations via HACC dialect
- [ ] Advanced operator fusion via HFusion
- [ ] Pipeline optimization and memory layout tuning
- [ ] Hardware testing and benchmarking

##### Advantages of MLIR Approach

- Leverages mature MLIR infrastructure for optimizations
- Enables gradual lowering through multiple passes
- Provides standardized interfaces for AI framework integration
- Supports cross-platform retargeting to other MLIR-supported accelerators
- Better integration with compiler toolchains

##### Testing

- ✅ **Unit tests**: `src/codegen/mlir/to_mlir.rs` - Comprehensive type conversion tests
- ✅ **Integration tests**: `tests/mlir/` - 14 passing tests covering core language features
- ✅ **Example programs**: `examples/core/` - Working examples demonstrating MLIR generation
- ✅ **Test coverage**: Constants, arithmetic, control flow, memory operations, GPU memory spaces

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

- **mlir/**: MLIR backend (primary) - Complete type conversion to MLIR, MLIR builder, AscendNPU-IR dialect bindings for Ascend NPUs
- **cuda/**: CUDA backend (experimental) - Data types for CUDA AST, translates Descend AST to CUDA AST, printing of the CUDA AST to C++ code
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
- `core/`: Core language examples for MLIR backend development

### AscendNPU-IR/

- Submodule containing Huawei's AscendNPU-IR MLIR dialect definitions
- HIVM, HACC, HFusion, and other Ascend-specific dialects
- Build tools and documentation for Ascend NPU compilation
- Integration tests and E2E use cases
