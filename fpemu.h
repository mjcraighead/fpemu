// fpemu (https://github.com/mjcraighead/fpemu)
// Copyright (c) 2025-2026 Matt Craighead
// SPDX-License-Identifier: MIT

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FPEMU_MXCSR_IE 0x0001 // invalid
#define FPEMU_MXCSR_DE 0x0002 // denormal operand
#define FPEMU_MXCSR_ZE 0x0004 // divide by zero
#define FPEMU_MXCSR_OE 0x0008 // overflow
#define FPEMU_MXCSR_UE 0x0010 // underflow
#define FPEMU_MXCSR_PE 0x0020 // precision (inexact)
#define FPEMU_MXCSR_DAZ 0x0040
#define FPEMU_MXCSR_EXCEPTION_MASKS 0x1F80
#define FPEMU_MXCSR_RM_NEAREST 0x0000
#define FPEMU_MXCSR_RM_DOWN 0x2000
#define FPEMU_MXCSR_RM_UP 0x4000
#define FPEMU_MXCSR_RM_ZERO 0x6000
#define FPEMU_MXCSR_RM_MASK 0x6000
#define FPEMU_MXCSR_FTZ 0x8000
#define FPEMU_MXCSR_DEFAULT 0x1F80
#define FPEMU_MXCSR_DEFINED_MASK 0xFFFF

#define FPEMU_EFLAGS_CF 0x0001
#define FPEMU_EFLAGS_PF 0x0004
#define FPEMU_EFLAGS_AF 0x0010
#define FPEMU_EFLAGS_ZF 0x0040
#define FPEMU_EFLAGS_SF 0x0080
#define FPEMU_EFLAGS_OF 0x0800
#define FPEMU_EFLAGS_COMI_MASK \
    (FPEMU_EFLAGS_CF | FPEMU_EFLAGS_PF | FPEMU_EFLAGS_AF | FPEMU_EFLAGS_ZF | \
     FPEMU_EFLAGS_SF | FPEMU_EFLAGS_OF)

// Operands and results are raw IEEE bit patterns. p_mxcsr is an in/out MXCSR value
// and must be non-NULL, have no bits outside FPEMU_MXCSR_DEFINED_MASK, and have all
// FPEMU_MXCSR_EXCEPTION_MASKS bits set. Immediate operands must be in architectural
// range: ROUND <= 15, CMP <= 31, VCVTPS2PH <= 7. CMP returns the scalar compare
// result bit pattern: all-ones for true, zero for false.
uint32_t fpemu_fp32_add(uint32_t a, uint32_t b, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_sub(uint32_t a, uint32_t b, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_mul(uint32_t a, uint32_t b, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_div(uint32_t a, uint32_t b, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_min(uint32_t a, uint32_t b, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_max(uint32_t a, uint32_t b, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_sqrt(uint32_t a, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_round(uint32_t a, uint32_t imm, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_cmp(uint32_t a, uint32_t b, uint32_t imm, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_comi(uint32_t a, uint32_t b, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_ucomi(uint32_t a, uint32_t b, uint32_t *p_mxcsr);

// FMA APIs model logical a*b +/- c with NaN precedence matching FMA3 132/231 encodings.
// To model a 213 encoding for the same logical expression, call with a and b swapped.
uint32_t fpemu_fp32_fmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_fmsub(uint32_t a, uint32_t b, uint32_t c, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_fnmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_fnmsub(uint32_t a, uint32_t b, uint32_t c, uint32_t *p_mxcsr);

uint64_t fpemu_fp64_add(uint64_t a, uint64_t b, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_sub(uint64_t a, uint64_t b, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_mul(uint64_t a, uint64_t b, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_div(uint64_t a, uint64_t b, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_min(uint64_t a, uint64_t b, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_max(uint64_t a, uint64_t b, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_sqrt(uint64_t a, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_round(uint64_t a, uint32_t imm, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_cmp(uint64_t a, uint64_t b, uint32_t imm, uint32_t *p_mxcsr);
uint32_t fpemu_fp64_comi(uint64_t a, uint64_t b, uint32_t *p_mxcsr);
uint32_t fpemu_fp64_ucomi(uint64_t a, uint64_t b, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_fmadd(uint64_t a, uint64_t b, uint64_t c, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_fmsub(uint64_t a, uint64_t b, uint64_t c, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_fnmadd(uint64_t a, uint64_t b, uint64_t c, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_fnmsub(uint64_t a, uint64_t b, uint64_t c, uint32_t *p_mxcsr);

uint32_t fpemu_fp64_to_fp32(uint64_t a, uint32_t *p_mxcsr);
uint64_t fpemu_fp32_to_fp64(uint32_t a, uint32_t *p_mxcsr);
uint32_t fpemu_fp16_to_fp32(uint16_t a, uint32_t *p_mxcsr);
uint16_t fpemu_fp32_to_fp16(uint32_t a, uint32_t imm, uint32_t *p_mxcsr);

uint32_t fpemu_int32_to_fp32(int32_t a, uint32_t *p_mxcsr);
uint32_t fpemu_int64_to_fp32(int64_t a, uint32_t *p_mxcsr);
uint64_t fpemu_int32_to_fp64(int32_t a, uint32_t *p_mxcsr);
uint64_t fpemu_int64_to_fp64(int64_t a, uint32_t *p_mxcsr);

uint32_t fpemu_fp32_to_i32(uint32_t a, uint32_t *p_mxcsr);
uint32_t fpemu_fp32_to_i32_trunc(uint32_t a, uint32_t *p_mxcsr);
uint64_t fpemu_fp32_to_i64(uint32_t a, uint32_t *p_mxcsr);
uint64_t fpemu_fp32_to_i64_trunc(uint32_t a, uint32_t *p_mxcsr);

uint32_t fpemu_fp64_to_i32(uint64_t a, uint32_t *p_mxcsr);
uint32_t fpemu_fp64_to_i32_trunc(uint64_t a, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_to_i64(uint64_t a, uint32_t *p_mxcsr);
uint64_t fpemu_fp64_to_i64_trunc(uint64_t a, uint32_t *p_mxcsr);

#ifdef __cplusplus
}
#endif
