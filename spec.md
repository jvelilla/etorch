**Here is the complete, ready-to-use specification document for eTorch.**

You can copy-paste this directly into your new repo as `SPECIFICATION.md` (or `docs/ARCHITECTURE.md`). It is written as the authoritative living spec that will guide the extraction from MicroGPT/v1 and all future development.

```markdown
# eTorch — Eiffel Torch Specification
**Version 1.0** — February 2026  
**Repository**: https://github.com/jvelilla/etorch  
**Extracted from**: MicroGPT v1 branch (https://github.com/jvelilla/MicroGPT/tree/v1)

## 1. Vision & Goals

eTorch is a **production-grade, native Eiffel deep-learning framework** that provides the same developer experience as PyTorch while being fundamentally safer, more maintainable, and contract-driven.

**Core promise**  
> “If the contracts pass, the tensor math is mathematically correct.”

### 1.1 Key Design Principles
| Principle              | Eiffel Advantage                          | PyTorch Mapping                  |
|------------------------|-------------------------------------------|----------------------------------|
| **Design by Contract** | `require` / `ensure` / `invariant` everywhere | Runtime shape/dtype checks       |
| **Ease of Use**        | Fluent, readable, IDE-friendly            | `torch.Tensor` API               |
| **Zero-cost abstractions** | Expanded types, once routines, agents | C++ templates / Python bindings |
| **Performance**        | Strided views + OpenBLAS + future GPU     | ATen + CUDA                      |
| **Debuggability**      | Contracts + EiffelStudio debugger         | TorchScript / pdb                |

## 2. Repository Structure (proposed)

```text
etorch/
├── src/
│   ├── core/
│   │   ├── ET_DEVICE.e
│   │   ├── ET_DTYPE.e
│   │   ├── ET_STORAGE.e          (abstract + REAL_64, INT32, etc.)
│   │   └── ET_TENSOR.e
│   ├── autograd/
│   │   ├── ET_VALUE.e
│   │   ├── ET_FUNCTION.e
│   │   └── ET_AUTOGRAD.e
│   ├── nn/
│   │   ├── ET_MODULE.e
│   │   ├── ET_PARAMETER.e
│   │   ├── ET_LINEAR.e
│   │   ├── ET_MULTIHEAD_ATTENTION.e
│   │   ├── ET_LAYER_NORM.e
│   │   ├── ET_TRANSFORMER_BLOCK.e
│   │   └── ET_SEQUENTIAL.e
│   ├── optim/
│   │   ├── ET_OPTIMIZER.e
│   │   └── ET_ADAM.e
│   ├── utils/
│   │   ├── ET_SAVE_LOAD.e
│   │   └── ET_FUNCTIONAL.e
│   └── ET_TORCH.e                (main entry point — torch. equivalent)
├── tests/
│   └── ET_TORCH_USE_CASES.e      (PyTorch parity tests)
├── docs/
│   └── SPECIFICATION.md          (this file)
├── lib/                          (OpenBLAS wrappers when ready)
└── etorch.ecf
```

## 3. Core Elements

### 3.1 ET_TORCH (main namespace)
```eiffel
class ET_TORCH
feature -- Factory functions (PyTorch style)
    tensor (data: ARRAY [REAL_64]; shape: ARRAY [INTEGER_32]): ET_TENSOR
    zeros (shape: ARRAY [INTEGER_32]): ET_TENSOR
    ones (shape: ARRAY [INTEGER_32]): ET_TENSOR
    randn (shape: ARRAY [INTEGER_32]): ET_TENSOR
    arange (start, stop, step: REAL_64): ET_TENSOR

feature -- Context managers
    no_grad (action: PROCEDURE): detachable ET_VALUE
        -- require: action /= Void
        -- ensure: gradients disabled during action
end
```

### 3.2 ET_TENSOR — The Heart of eTorch
**Current status from v1** (already excellent):
- Strided views (`view`, `reshape`, `transpose`, `narrow`)
- Broadcasting (dynamic, with full DbC)
- Polymorphic storage (`REAL_64`, `INTEGER_32`, …)
- All arithmetic guarded by contracts

**Full public API (target)**

```eiffel
class ET_TENSOR [G -> NUMERIC]
inherit
    ANY
        redefine
            out
        end

create
    make_from_storage,
    make_zeros,
    make_ones,
    make_randn

feature -- Contract-rich interface
    shape: ARRAY [INTEGER_32]
    strides: ARRAY [INTEGER_32]
    dtype: ET_DTYPE
    device: ET_DEVICE
    requires_grad: BOOLEAN

    -- Arithmetic (all with strong contracts)
    infix "+" (other: ET_TENSOR [G]): ET_TENSOR [G]
        require
            same_shape_or_broadcastable: is_broadcastable (other)
        ensure
            result_shape_correct: Result.shape ~ broadcast_shape (Current, other)

    matmul (other: ET_TENSOR [G]): ET_TENSOR [G]
        require
            matmul_compatible: inner_dim = other.outer_dim
        ensure
            result_rank: Result.rank = rank + other.rank - 2

    view (new_shape: ARRAY [INTEGER_32]): ET_TENSOR [G]
        require
            view_legal: total_elements = product (new_shape)
        ensure
            zero_copy: storage = old storage   -- critical invariant

    reshape (new_shape: ARRAY [INTEGER_32]): ET_TENSOR [G]
    transpose (dim0, dim1: INTEGER_32): ET_TENSOR [G]
    narrow (dim, start, length: INTEGER_32): ET_TENSOR [G]

feature -- Autograd
    backward
        require
            requires_grad
        ensure
            grad_computed: grad /= Void

    grad: detachable ET_TENSOR [G]
    detach: ET_TENSOR [G]

invariant
    shape_consistent_with_storage: product (shape) = storage.count
    strides_valid: valid_strides (strides, shape)
    device_consistent
end
```

**DbC emphasis examples**
- Every shape mismatch → **immediate contract violation** (no cryptic runtime error 10 layers deep)
- `view` guarantees **zero-copy**
- `matmul` guarantees mathematical correctness

### 3.3 Autograd System (ET_VALUE + ET_FUNCTION)
**Current v1 status**: Already tensor-aware, tape-based graph.

**Target design** (PyTorch tape-style, Eiffel style):
```eiffel
deferred class ET_FUNCTION
feature
    forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
        deferred
    backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
        deferred
end

class ET_VALUE
feature
    data: ET_TENSOR
    grad: detachable ET_TENSOR
    grad_fn: detachable ET_FUNCTION
    parents: ARRAY [ET_VALUE]

    backward
        require
            is_leaf or else grad_fn /= Void
end
```

### 3.4 Neural Networks (torch.nn equivalent)

```eiffel
deferred class ET_MODULE
feature
    parameters: LIST [ET_PARAMETER]
    forward (x: ET_TENSOR): ET_TENSOR
        deferred

    train, eval
    to_device (d: ET_DEVICE)

    state_dict: HASH_TABLE [ET_TENSOR, STRING]
    load_state_dict (state: HASH_TABLE [ET_TENSOR, STRING])
end
```

All layers (`ET_LINEAR`, `ET_MULTIHEAD_ATTENTION`, `ET_LAYER_NORM`, etc.) inherit from `ET_MODULE` and implement full DbC on `forward`.

### 3.5 Optimizers (torch.optim)

```eiffel
deferred class ET_OPTIMIZER
feature
    step
    zero_grad
end

class ET_ADAM
    -- Already implemented in v1 — just moved + contracts added
end
```

## 4. Performance Roadmap

**Phase 1 (Q1 2026)**
- Extract + stabilize current v1 code
- Full OpenBLAS `cblas_dgemm` integration for all matmul/bmm (strided calls → zero copy)

**Phase 2**
- GPU backend via Eiffel-OpenCL or CUDA FFI
- `ET_DEVICE {CPU, CUDA}`

**Phase 3**
- `torch.compile`-like JIT via EiffelStudio C code generation

## 5. Testing Strategy

- `ET_TORCH_USE_CASES.e` runs **exact same microGPT.py test vectors** from Karpathy
- Every new operation must have a PyTorch parity test + contract verification

## 6. Migration Plan from MicroGPT/v1

1. **Week 1**: Create directory structure above + copy existing classes
2. **Week 2**: Rename/move `ET_TENSOR` → `src/core/ET_TENSOR.e`, add missing contracts
3. **Week 3**: Extract `ET_VALUE`, `ET_ADAM`, `ET_GPT` into proper packages
4. **Week 4**: Add `ET_TORCH` facade + `ET_MODULE` hierarchy
5. **Ongoing**: Replace all internal MicroGPT uses with `etorch` dependency

## 7. Contribution & Style Rules

- Every public routine **must** have `require` + `ensure`
- Use `once` for singletons (devices, dtypes)
- Agents for callbacks (DataLoader, hooks)
- No magic numbers — named constants + contracts
- 100 % AutoTest coverage for core tensor ops

---

**This document is the single source of truth.**

Next step (your choice):

1. “Approve this spec — start extraction” → I’ll give you the exact commit-by-commit plan
2. “Review ET_TENSOR section — make contracts even stronger”
3. “Design the full NN module hierarchy with code examples”
4. “Add BLAS integration chapter with cblas_dgemm wrapper”

Just say the word and we ship the first clean eTorch release together.

Let’s make Eiffel the most correct deep-learning language on the planet. 🚀
```

This document is **complete, actionable, and already reflects the excellent foundation you built in v1**.  

Ready when you are — what do we tackle first?