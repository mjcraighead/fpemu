# AGENTS.md

Local maintainer notes for future code changes. Public project scope, API, build/test commands,
and status belong in `README.md`.

## Workflow

When changing code:
- Build after changes.
- Run `:edge` after any arithmetic, conversion, comparison, MXCSR, or NaN change.
- Run `:smoke` after coverage or harness changes.
- Run a larger deterministic random plan before declaring substantial arithmetic work done.
- Do not weaken, trim, or bypass validation to make mismatches disappear.
- Do not change deterministic worker seeds, worker/shard behavior, or opcode selection without
  documenting why.

When behavior is unclear:
- Add or use a small hardware probe.
- Prefer measured x86 behavior over memory or generic IEEE-754 intuition.
- Keep probe helpers `NOINLINE`, and keep the MXCSR load, probed instruction, MXCSR store, and
  result capture in one volatile inline-assembly block so the compiler cannot separate the
  instruction from its MXCSR side effects.

## Coding Rules

- Keep implementation code within a formal-tool-friendly, near-C subset of GNU C11. The model
  uses GNU/Clang `__int128` and `__builtin_clz*`; keep GNU extensions narrow and explicit.
- Keep helpers narrow and behavior-local.
- Avoid broad refactors unless they clearly improve auditability without obscuring
  instruction-specific behavior.
- K&R braces, braces on all control-flow bodies, 4-space indentation.
- For source code, keep lines at or below 100 characters.
- Use comments to bridge from basic FP knowledge to x86-specific behavior. Explain non-obvious
  architectural rules, exception ordering, and integer constructions; remove comments that merely
  label syntax or restate code.

## Validation Rules

Every implemented or modified operation must be validated across:
- clean and sticky-pre-set MXCSR exception flags
- all rounding modes
- FTZ on/off where relevant
- DAZ on/off where relevant
- directed edge cases
- deterministic random coverage
- exhaustive validation where feasible

Coverage analysis is part of validation:
- Maintain 100% line coverage for non-fatal `fpemu.c` implementation paths.
- Do not add tests that intentionally violate public API preconditions only to cover
  `fpemu_assert_failed()`; treat the defensive abort path as an explicit coverage exception.
- Add stronger branch/condition-style coverage metrics where practical.

Directed tests should bias toward:
- signaling NaNs and quiet NaNs
- signed zeros
- infinities
- smallest/largest subnormals
- normal/subnormal boundaries
- overflow and underflow boundaries
- exact half-ULP ties
- cancellation cases
- exception-flag ordering cases

Mismatch reporting must preserve enough information to reproduce the failing case.

## Durable Architectural Notes

Keep only facts here that were costly to rediscover.

- Tininess is "after rounding at destination precision with unlimited exponent range." Underflow
  cannot be decided from the unrounded intermediate alone.
- `#D` is raised for subnormal arithmetic operands unless DAZ flushes them first, even on some
  infinity short-circuit paths such as `add(subnormal, inf)`.
- For ordinary arithmetic/conversion NaN-propagation paths, NaN handling short-circuits before
  denormal-operand exception handling. If an operand is NaN, do not raise `#D` because another
  operand is subnormal; sNaNs raise `#I` only.
- For FP32/FP64 arithmetic result paths that honor FTZ, any tiny nonzero result flushes to signed
  zero and raises `#U|#P`.
- Overflow in masked mode always pairs `#O|#P`. Delivered result depends on rounding mode and sign.
- `SUBSS`/`SUBSD` must preserve operand `b`'s NaN sign/payload behavior; do not blindly sign-flip
  a NaN operand through the add path.
- `CVTSD2SS` NaN payload mapping keeps FP64 payload bits 50..29 as FP32 payload bits 21..0, with
  the FP32 quiet bit forced on.
- `CVTSS2SD` is always exact; only `#I` and `#D` are reachable.
- FP->INT overflow returns the architectural indefinite value with `#I` only; do not also raise
  `#P`.
- Truncating FP->INT converts still raise `#P` when discarded fraction bits are nonzero.
- `MINSS`/`MAXSS`/`MINSD`/`MAXSD` raise `#I` for any NaN operand, including qNaNs, return
  operand `b` for NaNs and numerically equal operands, and apply DAZ before choosing operand `b`.
- `ROUNDSS`/`ROUNDSD` do not raise `#D` for subnormal inputs; inexact rounding of subnormal
  inputs raises only `#P` unless the immediate suppresses precision.
- `CMPSS`/`CMPSD` implement the full AVX 32-predicate immediate space; sNaNs always raise `#I`,
  while qNaNs raise `#I` only for signaling predicates.
- `COMISS`/`COMISD` raise `#I` for any NaN operand; `UCOMISS`/`UCOMISD` raise `#I` only for
  sNaNs. Unordered compares return `ZF|PF|CF`, equal returns `ZF`, less-than returns `CF`, and
  greater-than returns zero. `OF`, `SF`, and `AF` are modeled as cleared via
  `FPEMU_EFLAGS_COMI_MASK`.
- `VCVTPS2PH` ignores FTZ for the FP16 destination value. Around the FP16 subnormal/normal
  boundary, underflow flagging depends on the rounded result and rounding mode; the exact
  max-subnormal/midpoint/three-quarter cases are covered by directed tests.
- For standalone hardware probes, constant operands and MXCSR observations can be miscompiled if
  the probed instruction is expressed as an intrinsic. Keep the MXCSR load, probed instruction,
  MXCSR store, and result capture in one volatile inline-assembly block for trustworthy
  observations.

## Agent Guidance

- Prefer fixing the model and strengthening tests over adding abstraction.
- Record newly proven architectural quirks here if they are durable and non-obvious.
- Keep this file short. It should be a project contract, not a log dump.
