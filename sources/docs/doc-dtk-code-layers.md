---
id: doc-dtk-code-layers
title: PTX, LLVM IR, target assembly and code objects describe different compiler layers
type: source-doc
architectures: []
tags: [assembly, triton]
date: '2026-10-08'
url: https://docs.nvidia.com/cuda/parallel-thread-execution/
confidence: source-reported
---

[NVIDIA PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/) defines a virtual instruction set that is translated to target GPU instructions.
[LLVM AMDGPU](https://llvm.org/docs/AMDGPUUsage.html) separately describes target LLVM IR/intrinsics, assembly and code-object interfaces.
These layers should not be collapsed into a generic label such as GPU assembly.

| Role | NVIDIA terminology | Observed DTK route, qualified by the experiment below |
|---|---|---|
| Compiler IR | LLVM/NVVM IR on LLVM-based routes | LLVM IR with target attributes/intrinsics |
| Virtual ISA | PTX | No separate PTX stage in the captured standard HCU backend |
| Target instruction assembly | SASS | gfx938 ISA text, recorded under amdgcn |
| Loadable device object | cubin | HSACO ELF code object |

The local Hygon facts belong to exp-argmax-bf16-inline-20261008, which captures the installed HCU backend stage definition.
The upstream AMDGPU documentation is not evidence that upstream LLVM supports gfx938 or that Hygon is an AMD vendor target.
HSACO is a code-object format, not a source language. The toolchain name amdgcn does not select a different architecture on the caller's behalf.

[Triton inline_asm_elementwise](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html) embeds an assembly template with constraints, result dtype, purity and element packing.
For target-specific register constraints, [LLVM inline assembly rules](https://llvm.org/docs/LangRef.html#inline-assembler-expressions) remain relevant.
An inline template can request a register class while LLVM still chooses physical registers; it does not imply a separate virtual ISA or a fully handwritten code object.

Purity describes side effects. It does not state that an arbitrary assembly block is an unsigned maximum, is associative, or admits the same algebraic rewrites as a recognized IR operation.
Use the current installed API and final target artifacts: the latest online Gluon API can contain interfaces absent from the vendor package.
The linked experiment tests only a pure register-only uint32 maximum, not memory descriptors, barriers, packed matrix operands or arbitrary inline kernels.
