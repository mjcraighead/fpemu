# fpemu

`fpemu` is a small software reference model for AVX2-era scalar x86 floating-point
operations. It is intended to be bit-exact and audit-friendly: the software path
uses integer and bit-level logic, not host floating-point arithmetic, and the test
harness compares result bits and final MXCSR against hardware.

The model and its test harness are GNU C11, using GNU/Clang `__int128` and
`__builtin_clz*`, with no host floating point in the model. The harness uses x86
inline assembly for hardware probing.

This is not a fast math library, a compiler abstraction layer, or a general replacement
for Berkeley SoftFloat. The goal is narrower: executable reference semantics for scalar
x86 floating-point instructions, including the architectural details that are easy to
lose when using a generic IEEE-754 library directly.

## Scope

Implemented operations currently include:
- FP32 and FP64 scalar `ADD`, `SUB`, `MUL`, `DIV`, `SQRT`, `MIN`, `MAX`, `ROUND`, and `CMP`
- FP32 and FP64 scalar compare-to-flags operations: `COMISS`, `UCOMISS`, `COMISD`, and `UCOMISD`
- FP32 and FP64 scalar FMA-family operations: `FMADD`, `FMSUB`, `FNMADD`, and `FNMSUB`
- FP32 <-> FP64 scalar conversions
- F16C scalar-lane FP16 <-> FP32 conversions
- INT32/INT64 -> FP32/FP64 scalar conversions
- FP32/FP64 -> INT32/INT64 scalar conversions, including truncating forms

The observable result for each modeled operation is:
- the destination value bits
- for compare-to-flags operations, the defined `CF`, `PF`, `ZF`, `AF`, `SF`, and `OF` EFLAGS bits
- the final MXCSR sticky exception flags and control bits

Masked exception behavior is in scope. Trapping behavior is not modeled.

`RCPSS` and `RSQRTSS` are excluded from the first release. These instructions are
architecturally approximate and have been observed to produce different result bits on
Intel and AMD CPUs, so there is no single cross-vendor bit-exact result for this
reference model to target. They can be revisited later as CPU-profile-specific models
if that becomes useful.

Packed/vector instruction modeling is intentionally out of scope. Different downstream
users may want different register layouts, lane containers, memory forms, merge/zeroing
rules, instruction-decode boundaries, or integration points with larger emulators or
verification systems.

`fpemu` keeps the public model at the numeric primitive level: scalar value bits and
MXCSR in, scalar value bits and MXCSR out. Callers can build packed semantics by
applying scalar operations lane by lane while reusing the same MXCSR, naturally
accumulating sticky exception flags.

## Public API

`fpemu.h` exposes `fpemu_*` functions plus `FPEMU_MXCSR_*` and `FPEMU_EFLAGS_*` bit
definitions. Operands and results are raw IEEE bit patterns stored in fixed-width
integer types. Each function takes a non-NULL `uint32_t *p_mxcsr` pointing to an
in/out MXCSR image.

Callers normally initialize `*p_mxcsr` to `FPEMU_MXCSR_DEFAULT` plus any desired
rounding mode, DAZ/FTZ controls, and pre-existing sticky exception bits. The model
implements masked exception behavior only: `*p_mxcsr` must have no bits outside
`FPEMU_MXCSR_DEFINED_MASK`, and all bits in `FPEMU_MXCSR_EXCEPTION_MASKS` must be
set. The operation ORs in newly raised masked exception bits, matching the sticky
behavior of hardware MXCSR.

Scalar `CMP` operations return the destination value bit pattern: all ones for true,
zero for false. Compare-to-flags operations return the relevant EFLAGS bits masked by
`FPEMU_EFLAGS_COMI_MASK`. FP-to-int conversions return the raw integer-result bits as
`uint32_t` or `uint64_t`, including architectural indefinite values.

Minimal example:
```c
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include "fpemu.h"

int main(void) {
    uint32_t mxcsr = FPEMU_MXCSR_DEFAULT;
    uint32_t result = fpemu_fp32_add(0x3F800000, 0x40000000, &mxcsr);
    printf("result=0x%" PRIX32 " mxcsr=0x%" PRIX32 "\n", result, mxcsr);
    return 0;
}
```

Compile it directly with:
```sh
gcc -std=gnu11 -O2 -Wall -Wextra -Werror example.c fpemu.c -o example
```

## Correctness Model

The model aims for literal architectural equivalence with x86 hardware for the
implemented scalar operations:
- exact result bits
- exact final MXCSR
- signed zeros
- infinities
- signaling and quiet NaNs
- NaN quieting, payload, and sign behavior
- subnormal inputs and outputs
- DAZ and FTZ
- all MXCSR rounding modes
- sticky exception flags
- overflow, underflow, inexact, denormal-operand, divide-by-zero, and invalid behavior

The implementation keeps arithmetic explicit and machine-independent with fixed-width
integer operations, masks, shifts, bit scans, and small local helpers.

## Why This Exists

I wrote `fpemu` because I wanted to fully understand x86 floating point and to have a
simple, auditable reference model I could consult for each operation.

SoftFloat is the standard high-quality software floating-point library, and it is an
excellent foundation for many projects. `fpemu` is focused on a narrower problem:
scalar x86 instruction behavior as observed through result bits and MXCSR. That includes
DAZ, FTZ, MXCSR rounding and sticky flags, instruction-specific NaN behavior, conversion
indefinite values, and exception ordering.

The code is intentionally compact. A user should be able to inspect the relevant
operation and its tests without first learning a large emulator or general
floating-point library.

## Repository Layout
- `fpemu.c`: software reference implementation
- `fpemu.h`: public declarations and `FPEMU_*` bit definitions
- `test.c`: hardware probes, directed edge tests, randomized validation, CLI
- `rules.py`: build and validation targets for `make.py`
- `make.py`: vendored build driver

Generated files are written under `_out/`.

## Building

Use `make.py`:
```sh
./make.py :build
```

The default build uses `gcc` with `-std=gnu11 -O2 -Wall -Wextra -Werror -march=x86-64-v3`.
The model itself does not use x86-specific intrinsics or inline assembly, but the
hardware validation harness uses inline assembly probes and reads/writes MXCSR, so it
should be run on x86-64-v3-capable hardware.

## Validation

Core targets:
```sh
./make.py :edge
./make.py :smoke
./make.py :test
./make.py :long
./make.py :exhaustive
```

What they do:
- `:edge`: directed edge-case suite across operation classes, rounding modes, FTZ/DAZ
  settings, and clean/sticky MXCSR states
- `:smoke`: directed edge suite plus a short deterministic randomized run
- `:test`: standard deterministic randomized validation: 256 workers x 10^8 cases
- `:long`: longer deterministic randomized soak: 256 workers x 10^10 cases
- `:exhaustive`: exhaustive validation for implemented single-data-input operations
  whose data input space is at most 2^32 values, over the relevant MXCSR controls
  and immediate bits

The test binary can also be invoked directly:
```sh
_out/fpemu edge
_out/fpemu rand <log10-cases> <seed>
_out/fpemu exhaustive <op> <rm> <daz> <ftz> <imm> <index> <count>
```

Each command also accepts an optional final log path, used by `make.py` targets as a
small status artifact. Randomized tests are deterministic for a given seed; the
worker-split `make.py` targets use fixed seeds 1..256. `rand` accepts
`log10-cases` values 0..20. `exhaustive` accepts shard counts 1..10000, with
`index < count`. Mismatch reports print enough operand, MXCSR, and operation
information to reproduce the failing case.

## Coverage

Coverage is measured with `llvm-cov` through the coverage targets:
```sh
./make.py :coverage-build
./make.py :coverage-report
./make.py :coverage-test-report
```

The report targets build their required profile artifacts. Individual coverage logs
and `.profraw` files can also be requested directly by path when debugging.

The coverage rules currently expect `clang-21`, `llvm-profdata-21`, and `llvm-cov-21`
in `PATH`.

The model coverage expectation is full coverage of non-fatal implementation paths in
`fpemu.c`. The defensive `FPEMU_ASSERT` abort path is intentionally not exercised by
validation, because public entry points require valid preconditions.

## Status

`fpemu` is usable as a first-release scalar reference model for the operations listed
above. The public API is intended to be small and stable, and the implemented operations
are hardware-compared by directed and deterministic randomized tests.

Validation is still expected to grow. Current coverage includes directed edge tests,
randomized worker-split runs, exhaustive testing for tractable single-input operations,
and `llvm-cov` reporting. Testing has been run on an Intel Core i7-4770S and an AMD
Ryzen 9 9950X.

Pre-release fault-injection checks inserted 25 intentional implementation bugs across
NaN handling, MXCSR flags, rounding, conversions, comparisons, and FMA behavior. The
directed edge suite caught all 25; the randomized smoke plan caught 24/25, and the
full randomized `:test` plan caught the remaining case.

## License
Copyright © 2025-2026 Matt Craighead

Released under the terms of the MIT License — see [LICENSE](LICENSE) for details.
