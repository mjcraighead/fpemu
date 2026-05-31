// fpemu (https://github.com/mjcraighead/fpemu)
// Copyright (c) 2025-2026 Matt Craighead
// SPDX-License-Identifier: MIT

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include "fpemu.h"

#define INNER_LOOPS 1000
#define MAX_PRINT 100

#define NORETURN __attribute__((noreturn))
#define NOINLINE __attribute__((noinline))

#define TEST_ASSERT(cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            test_assert_failed(__func__, fmt, ##__VA_ARGS__); \
        } \
    } while (0)

static NORETURN void test_assert_failed(const char *function, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "test assertion failed: %s: ", function);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

static __uint128_t s_lehmer_state = 1;
static bool s_quiet_success = false;

static void lehmer64_seed(uint64_t seed) {
    // Mix the user-visible seed into the high state bits and force the low bits nonzero;
    // Lehmer64 would stay at zero forever from a zero state.
    uint64_t m = seed * 0x9E3779B97F4A7C15ull + 1;
    s_lehmer_state = ((__uint128_t)m << 64) | (uint64_t)1;
}

static uint64_t lehmer64(void) {
    s_lehmer_state *= 0xDA942042E4DD58B5ull;
    return s_lehmer_state >> 64;
}

// Hardware probes are kept noinline so each instruction has a stable wrapper in
// disassembly. The MXCSR load, probed instruction, MXCSR store, and result move
// live in one volatile asm block so compiler scheduling cannot separate the
// instruction from its architectural MXCSR side effects.
#define HW_FP32_BINOP(name, opcode) \
    static NOINLINE uint32_t hw_fp32_##name(uint32_t a, uint32_t b, uint32_t *p_mxcsr) { \
        uint32_t ret; \
        uint32_t in_mxcsr = *p_mxcsr; \
        uint32_t out_mxcsr; \
        asm volatile("vmovd %[a], %%xmm0\n\t" \
                     "vmovd %[b], %%xmm1\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     #opcode " %%xmm1, %%xmm0, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovd %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [b] "rm"(b), [in] "m"(in_mxcsr) \
                     : "xmm0", "xmm1", "memory", "cc"); \
        *p_mxcsr = out_mxcsr; \
        return ret; \
    }

#define HW_FP64_BINOP(name, opcode) \
    static NOINLINE uint64_t hw_fp64_##name(uint64_t a, uint64_t b, uint32_t *p_mxcsr) { \
        uint64_t ret; \
        uint32_t in_mxcsr = *p_mxcsr; \
        uint32_t out_mxcsr; \
        asm volatile("vmovq %[a], %%xmm0\n\t" \
                     "vmovq %[b], %%xmm1\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     #opcode " %%xmm1, %%xmm0, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovq %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [b] "rm"(b), [in] "m"(in_mxcsr) \
                     : "xmm0", "xmm1", "memory", "cc"); \
        *p_mxcsr = out_mxcsr; \
        return ret; \
    }

#define HW_FP32_FMA(name, opcode) \
    static NOINLINE uint32_t hw_fp32_##name(uint32_t a, uint32_t b, uint32_t c, \
                                            uint32_t *p_mxcsr) { \
        uint32_t ret; \
        uint32_t in_mxcsr = *p_mxcsr; \
        uint32_t out_mxcsr; \
        asm volatile("vmovd %[a], %%xmm0\n\t" \
                     "vmovd %[b], %%xmm1\n\t" \
                     "vmovd %[c], %%xmm2\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     #opcode " %%xmm1, %%xmm2, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovd %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [b] "rm"(b), [c] "rm"(c), [in] "m"(in_mxcsr) \
                     : "xmm0", "xmm1", "xmm2", "memory", "cc"); \
        *p_mxcsr = out_mxcsr; \
        return ret; \
    }

#define HW_FP64_FMA(name, opcode) \
    static NOINLINE uint64_t hw_fp64_##name(uint64_t a, uint64_t b, uint64_t c, \
                                            uint32_t *p_mxcsr) { \
        uint64_t ret; \
        uint32_t in_mxcsr = *p_mxcsr; \
        uint32_t out_mxcsr; \
        asm volatile("vmovq %[a], %%xmm0\n\t" \
                     "vmovq %[b], %%xmm1\n\t" \
                     "vmovq %[c], %%xmm2\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     #opcode " %%xmm1, %%xmm2, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovq %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [b] "rm"(b), [c] "rm"(c), [in] "m"(in_mxcsr) \
                     : "xmm0", "xmm1", "xmm2", "memory", "cc"); \
        *p_mxcsr = out_mxcsr; \
        return ret; \
    }

#define HW_FP32_UNARY(name, opcode) \
    static NOINLINE uint32_t hw_fp32_##name(uint32_t a, uint32_t *p_mxcsr) { \
        uint32_t ret; \
        uint32_t in_mxcsr = *p_mxcsr; \
        uint32_t out_mxcsr; \
        asm volatile("vmovd %[a], %%xmm0\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     #opcode " %%xmm0, %%xmm0, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovd %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [in] "m"(in_mxcsr) \
                     : "xmm0", "memory", "cc"); \
        *p_mxcsr = out_mxcsr; \
        return ret; \
    }

#define HW_FP64_UNARY(name, opcode) \
    static NOINLINE uint64_t hw_fp64_##name(uint64_t a, uint32_t *p_mxcsr) { \
        uint64_t ret; \
        uint32_t in_mxcsr = *p_mxcsr; \
        uint32_t out_mxcsr; \
        asm volatile("vmovq %[a], %%xmm0\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     #opcode " %%xmm0, %%xmm0, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovq %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [in] "m"(in_mxcsr) \
                     : "xmm0", "memory", "cc"); \
        *p_mxcsr = out_mxcsr; \
        return ret; \
    }

HW_FP32_BINOP(add, vaddss)
HW_FP32_BINOP(sub, vsubss)
HW_FP32_BINOP(mul, vmulss)
HW_FP32_BINOP(div, vdivss)
HW_FP32_BINOP(min, vminss)
HW_FP32_BINOP(max, vmaxss)
HW_FP32_UNARY(sqrt, vsqrtss)
HW_FP32_FMA(fmadd, vfmadd132ss)
HW_FP32_FMA(fmsub, vfmsub132ss)
HW_FP32_FMA(fnmadd, vfnmadd132ss)
HW_FP32_FMA(fnmsub, vfnmsub132ss)

HW_FP64_BINOP(add, vaddsd)
HW_FP64_BINOP(sub, vsubsd)
HW_FP64_BINOP(mul, vmulsd)
HW_FP64_BINOP(div, vdivsd)
HW_FP64_BINOP(min, vminsd)
HW_FP64_BINOP(max, vmaxsd)
HW_FP64_UNARY(sqrt, vsqrtsd)
HW_FP64_FMA(fmadd, vfmadd132sd)
HW_FP64_FMA(fmsub, vfmsub132sd)
HW_FP64_FMA(fnmadd, vfnmadd132sd)
HW_FP64_FMA(fnmsub, vfnmsub132sd)

#define HW_FP32_CMP_CASE(i) \
    case i: \
        asm volatile("vmovd %[a], %%xmm0\n\t" \
                     "vmovd %[b], %%xmm1\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     "vcmpss $" #i ", %%xmm1, %%xmm0, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovd %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [b] "rm"(b), [in] "m"(in_mxcsr) \
                     : "xmm0", "xmm1", "memory", "cc"); \
        break;

static NOINLINE uint32_t hw_fp32_cmp(uint32_t a, uint32_t b, uint32_t imm, uint32_t *p_mxcsr) {
    uint32_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    switch (imm & 31) {
        HW_FP32_CMP_CASE(0)  HW_FP32_CMP_CASE(1)  HW_FP32_CMP_CASE(2)  HW_FP32_CMP_CASE(3)
        HW_FP32_CMP_CASE(4)  HW_FP32_CMP_CASE(5)  HW_FP32_CMP_CASE(6)  HW_FP32_CMP_CASE(7)
        HW_FP32_CMP_CASE(8)  HW_FP32_CMP_CASE(9)  HW_FP32_CMP_CASE(10) HW_FP32_CMP_CASE(11)
        HW_FP32_CMP_CASE(12) HW_FP32_CMP_CASE(13) HW_FP32_CMP_CASE(14) HW_FP32_CMP_CASE(15)
        HW_FP32_CMP_CASE(16) HW_FP32_CMP_CASE(17) HW_FP32_CMP_CASE(18) HW_FP32_CMP_CASE(19)
        HW_FP32_CMP_CASE(20) HW_FP32_CMP_CASE(21) HW_FP32_CMP_CASE(22) HW_FP32_CMP_CASE(23)
        HW_FP32_CMP_CASE(24) HW_FP32_CMP_CASE(25) HW_FP32_CMP_CASE(26) HW_FP32_CMP_CASE(27)
        HW_FP32_CMP_CASE(28) HW_FP32_CMP_CASE(29) HW_FP32_CMP_CASE(30)
        default: HW_FP32_CMP_CASE(31);
    }
    *p_mxcsr = out_mxcsr;
    return ret;
}

#define HW_FP64_CMP_CASE(i) \
    case i: \
        asm volatile("vmovq %[a], %%xmm0\n\t" \
                     "vmovq %[b], %%xmm1\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     "vcmpsd $" #i ", %%xmm1, %%xmm0, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovq %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [b] "rm"(b), [in] "m"(in_mxcsr) \
                     : "xmm0", "xmm1", "memory", "cc"); \
        break;

static NOINLINE uint64_t hw_fp64_cmp(uint64_t a, uint64_t b, uint32_t imm, uint32_t *p_mxcsr) {
    uint64_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    switch (imm & 31) {
        HW_FP64_CMP_CASE(0)  HW_FP64_CMP_CASE(1)  HW_FP64_CMP_CASE(2)  HW_FP64_CMP_CASE(3)
        HW_FP64_CMP_CASE(4)  HW_FP64_CMP_CASE(5)  HW_FP64_CMP_CASE(6)  HW_FP64_CMP_CASE(7)
        HW_FP64_CMP_CASE(8)  HW_FP64_CMP_CASE(9)  HW_FP64_CMP_CASE(10) HW_FP64_CMP_CASE(11)
        HW_FP64_CMP_CASE(12) HW_FP64_CMP_CASE(13) HW_FP64_CMP_CASE(14) HW_FP64_CMP_CASE(15)
        HW_FP64_CMP_CASE(16) HW_FP64_CMP_CASE(17) HW_FP64_CMP_CASE(18) HW_FP64_CMP_CASE(19)
        HW_FP64_CMP_CASE(20) HW_FP64_CMP_CASE(21) HW_FP64_CMP_CASE(22) HW_FP64_CMP_CASE(23)
        HW_FP64_CMP_CASE(24) HW_FP64_CMP_CASE(25) HW_FP64_CMP_CASE(26) HW_FP64_CMP_CASE(27)
        HW_FP64_CMP_CASE(28) HW_FP64_CMP_CASE(29) HW_FP64_CMP_CASE(30)
        default: HW_FP64_CMP_CASE(31);
    }
    *p_mxcsr = out_mxcsr;
    return ret;
}

#define HW_FP32_ROUND_CASE(i) \
    case i: \
        asm volatile("vmovd %[a], %%xmm0\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     "vroundss $" #i ", %%xmm0, %%xmm0, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovd %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [in] "m"(in_mxcsr) \
                     : "xmm0", "memory", "cc"); \
        break;

static NOINLINE uint32_t hw_fp32_round(uint32_t a, uint32_t imm, uint32_t *p_mxcsr) {
    uint32_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    switch (imm & 15) {
        HW_FP32_ROUND_CASE(0)  HW_FP32_ROUND_CASE(1)  HW_FP32_ROUND_CASE(2)  HW_FP32_ROUND_CASE(3)
        HW_FP32_ROUND_CASE(4)  HW_FP32_ROUND_CASE(5)  HW_FP32_ROUND_CASE(6)  HW_FP32_ROUND_CASE(7)
        HW_FP32_ROUND_CASE(8)  HW_FP32_ROUND_CASE(9)  HW_FP32_ROUND_CASE(10) HW_FP32_ROUND_CASE(11)
        HW_FP32_ROUND_CASE(12) HW_FP32_ROUND_CASE(13) HW_FP32_ROUND_CASE(14)
        default: HW_FP32_ROUND_CASE(15);
    }
    *p_mxcsr = out_mxcsr;
    return ret;
}

#define HW_FP64_ROUND_CASE(i) \
    case i: \
        asm volatile("vmovq %[a], %%xmm0\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     "vroundsd $" #i ", %%xmm0, %%xmm0, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovq %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [in] "m"(in_mxcsr) \
                     : "xmm0", "memory", "cc"); \
        break;

static NOINLINE uint64_t hw_fp64_round(uint64_t a, uint32_t imm, uint32_t *p_mxcsr) {
    uint64_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    switch (imm & 15) {
        HW_FP64_ROUND_CASE(0)  HW_FP64_ROUND_CASE(1)  HW_FP64_ROUND_CASE(2)  HW_FP64_ROUND_CASE(3)
        HW_FP64_ROUND_CASE(4)  HW_FP64_ROUND_CASE(5)  HW_FP64_ROUND_CASE(6)  HW_FP64_ROUND_CASE(7)
        HW_FP64_ROUND_CASE(8)  HW_FP64_ROUND_CASE(9)  HW_FP64_ROUND_CASE(10) HW_FP64_ROUND_CASE(11)
        HW_FP64_ROUND_CASE(12) HW_FP64_ROUND_CASE(13) HW_FP64_ROUND_CASE(14)
        default: HW_FP64_ROUND_CASE(15);
    }
    *p_mxcsr = out_mxcsr;
    return ret;
}

#define HW_COMI(name, src_t, opcode, mov) \
    static NOINLINE uint32_t hw_##name(src_t a, src_t b, uint32_t *p_mxcsr) { \
        uint64_t flags; \
        uint32_t in_mxcsr = *p_mxcsr; \
        uint32_t out_mxcsr; \
        asm volatile(mov " %[a], %%xmm0\n\t" \
                    mov " %[b], %%xmm1\n\t" \
                    "vldmxcsr %[in]\n\t" \
                    #opcode " %%xmm1, %%xmm0\n\t" \
                    "pushfq\n\t" \
                    "popq %[flags]\n\t" \
                    "vstmxcsr %[out]" \
                    : [flags] "=r"(flags), [out] "=m"(out_mxcsr) \
                    : [a] "rm"(a), [b] "rm"(b), [in] "m"(in_mxcsr) \
                    : "xmm0", "xmm1", "memory", "cc"); \
        *p_mxcsr = out_mxcsr; \
        return (uint32_t)flags & FPEMU_EFLAGS_COMI_MASK; \
    }

HW_COMI(fp32_comi, uint32_t, vcomiss, "vmovd")
HW_COMI(fp32_ucomi, uint32_t, vucomiss, "vmovd")
HW_COMI(fp64_comi, uint64_t, vcomisd, "vmovq")
HW_COMI(fp64_ucomi, uint64_t, vucomisd, "vmovq")

static NOINLINE uint32_t hw_fp64_to_fp32(uint64_t a, uint32_t *p_mxcsr) {
    uint32_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    asm volatile("vmovq %[a], %%xmm0\n\t"
                 "vldmxcsr %[in]\n\t"
                 "vcvtsd2ss %%xmm0, %%xmm0, %%xmm0\n\t"
                 "vstmxcsr %[out]\n\t"
                 "vmovd %%xmm0, %[ret]"
                 : [ret] "=r"(ret), [out] "=m"(out_mxcsr)
                 : [a] "rm"(a), [in] "m"(in_mxcsr)
                 : "xmm0", "memory", "cc");
    *p_mxcsr = out_mxcsr;
    return ret;
}

static NOINLINE uint64_t hw_fp32_to_fp64(uint32_t a, uint32_t *p_mxcsr) {
    uint64_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    asm volatile("vmovd %[a], %%xmm0\n\t"
                 "vpxor %%xmm1, %%xmm1, %%xmm1\n\t"
                 "vldmxcsr %[in]\n\t"
                 "vcvtss2sd %%xmm0, %%xmm1, %%xmm0\n\t"
                 "vstmxcsr %[out]\n\t"
                 "vmovq %%xmm0, %[ret]"
                 : [ret] "=r"(ret), [out] "=m"(out_mxcsr)
                 : [a] "rm"(a), [in] "m"(in_mxcsr)
                 : "xmm0", "xmm1", "memory", "cc");
    *p_mxcsr = out_mxcsr;
    return ret;
}

static NOINLINE uint32_t hw_fp16_to_fp32(uint16_t a, uint32_t *p_mxcsr) {
    uint32_t ret;
    uint32_t a32 = a;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    asm volatile("vmovd %[a], %%xmm0\n\t"
                 "vldmxcsr %[in]\n\t"
                 "vcvtph2ps %%xmm0, %%xmm0\n\t"
                 "vstmxcsr %[out]\n\t"
                 "vmovd %%xmm0, %[ret]"
                 : [ret] "=r"(ret), [out] "=m"(out_mxcsr)
                 : [a] "rm"(a32), [in] "m"(in_mxcsr)
                 : "xmm0", "memory", "cc");
    *p_mxcsr = out_mxcsr;
    return ret;
}

#define HW_FP32_TO_FP16_CASE(i) \
    case i: \
        asm volatile("vmovd %[a], %%xmm0\n\t" \
                     "vldmxcsr %[in]\n\t" \
                     "vcvtps2ph $" #i ", %%xmm0, %%xmm0\n\t" \
                     "vstmxcsr %[out]\n\t" \
                     "vmovd %%xmm0, %[ret]" \
                     : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                     : [a] "rm"(a), [in] "m"(in_mxcsr) \
                     : "xmm0", "memory", "cc"); \
        break

static NOINLINE uint16_t hw_fp32_to_fp16(uint32_t a, uint32_t imm, uint32_t *p_mxcsr) {
    uint32_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    switch (imm & 7) {
        HW_FP32_TO_FP16_CASE(0); HW_FP32_TO_FP16_CASE(1); HW_FP32_TO_FP16_CASE(2);
        HW_FP32_TO_FP16_CASE(3); HW_FP32_TO_FP16_CASE(4); HW_FP32_TO_FP16_CASE(5);
        HW_FP32_TO_FP16_CASE(6);
        default: HW_FP32_TO_FP16_CASE(7);
    }
    *p_mxcsr = out_mxcsr;
    return (uint16_t)ret;
}

static NOINLINE uint32_t hw_int32_to_fp32(int32_t a, uint32_t *p_mxcsr) {
    uint32_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    asm volatile("vpxor %%xmm0, %%xmm0, %%xmm0\n\t"
                 "vldmxcsr %[in]\n\t"
                 "vcvtsi2ssl %[a], %%xmm0, %%xmm0\n\t"
                 "vstmxcsr %[out]\n\t"
                 "vmovd %%xmm0, %[ret]"
                 : [ret] "=r"(ret), [out] "=m"(out_mxcsr)
                 : [a] "rm"(a), [in] "m"(in_mxcsr)
                 : "xmm0", "memory", "cc");
    *p_mxcsr = out_mxcsr;
    return ret;
}

static NOINLINE uint32_t hw_int64_to_fp32(int64_t a, uint32_t *p_mxcsr) {
    uint32_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    asm volatile("vpxor %%xmm0, %%xmm0, %%xmm0\n\t"
                 "vldmxcsr %[in]\n\t"
                 "vcvtsi2ssq %[a], %%xmm0, %%xmm0\n\t"
                 "vstmxcsr %[out]\n\t"
                 "vmovd %%xmm0, %[ret]"
                 : [ret] "=r"(ret), [out] "=m"(out_mxcsr)
                 : [a] "rm"(a), [in] "m"(in_mxcsr)
                 : "xmm0", "memory", "cc");
    *p_mxcsr = out_mxcsr;
    return ret;
}

static NOINLINE uint64_t hw_int32_to_fp64(int32_t a, uint32_t *p_mxcsr) {
    uint64_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    asm volatile("vpxor %%xmm0, %%xmm0, %%xmm0\n\t"
                 "vldmxcsr %[in]\n\t"
                 "vcvtsi2sdl %[a], %%xmm0, %%xmm0\n\t"
                 "vstmxcsr %[out]\n\t"
                 "vmovq %%xmm0, %[ret]"
                 : [ret] "=r"(ret), [out] "=m"(out_mxcsr)
                 : [a] "rm"(a), [in] "m"(in_mxcsr)
                 : "xmm0", "memory", "cc");
    *p_mxcsr = out_mxcsr;
    return ret;
}

static NOINLINE uint64_t hw_int64_to_fp64(int64_t a, uint32_t *p_mxcsr) {
    uint64_t ret;
    uint32_t in_mxcsr = *p_mxcsr;
    uint32_t out_mxcsr;
    asm volatile("vpxor %%xmm0, %%xmm0, %%xmm0\n\t"
                 "vldmxcsr %[in]\n\t"
                 "vcvtsi2sdq %[a], %%xmm0, %%xmm0\n\t"
                 "vstmxcsr %[out]\n\t"
                 "vmovq %%xmm0, %[ret]"
                 : [ret] "=r"(ret), [out] "=m"(out_mxcsr)
                 : [a] "rm"(a), [in] "m"(in_mxcsr)
                 : "xmm0", "memory", "cc");
    *p_mxcsr = out_mxcsr;
    return ret;
}

#define HW_FP_TO_INT(name, src_t, dst_t, mov, opcode) \
static NOINLINE dst_t hw_##name(src_t a, uint32_t *p_mxcsr) { \
    dst_t ret; \
    uint32_t in_mxcsr = *p_mxcsr; \
    uint32_t out_mxcsr; \
    asm volatile(mov " %[a], %%xmm0\n\t" \
                 "vldmxcsr %[in]\n\t" \
                 #opcode " %%xmm0, %[ret]\n\t" \
                 "vstmxcsr %[out]" \
                 : [ret] "=r"(ret), [out] "=m"(out_mxcsr) \
                 : [a] "rm"(a), [in] "m"(in_mxcsr) \
                 : "xmm0", "memory", "cc"); \
    *p_mxcsr = out_mxcsr; \
    return ret; \
}

HW_FP_TO_INT(fp32_to_i32, uint32_t, uint32_t, "vmovd", vcvtss2sil)
HW_FP_TO_INT(fp32_to_i32_trunc, uint32_t, uint32_t, "vmovd", vcvttss2sil)
HW_FP_TO_INT(fp32_to_i64, uint32_t, uint64_t, "vmovd", vcvtss2siq)
HW_FP_TO_INT(fp32_to_i64_trunc, uint32_t, uint64_t, "vmovd", vcvttss2siq)
HW_FP_TO_INT(fp64_to_i32, uint64_t, uint32_t, "vmovq", vcvtsd2sil)
HW_FP_TO_INT(fp64_to_i32_trunc, uint64_t, uint32_t, "vmovq", vcvttsd2sil)
HW_FP_TO_INT(fp64_to_i64, uint64_t, uint64_t, "vmovq", vcvtsd2siq)
HW_FP_TO_INT(fp64_to_i64_trunc, uint64_t, uint64_t, "vmovq", vcvttsd2siq)

typedef enum test_op {
    OP_ADDSS,
    OP_SUBSS,
    OP_MULSS,
    OP_DIVSS,
    OP_MINSS,
    OP_MAXSS,
    OP_SQRTSS,
    OP_ROUNDSS,
    OP_CMPSS,
    OP_FMADDSS,
    OP_FMSUBSS,
    OP_FNMADDSS,
    OP_FNMSUBSS,
    OP_COMISS,
    OP_UCOMISS,
    OP_ADDSD,
    OP_SUBSD,
    OP_MULSD,
    OP_DIVSD,
    OP_MINSD,
    OP_MAXSD,
    OP_SQRTSD,
    OP_ROUNDSD,
    OP_CMPSD,
    OP_FMADDSD,
    OP_FMSUBSD,
    OP_FNMADDSD,
    OP_FNMSUBSD,
    OP_COMISD,
    OP_UCOMISD,
    OP_CVTSD2SS,
    OP_CVTSS2SD,
    OP_VCVTPH2PS_LANE,
    OP_VCVTPS2PH_LANE,
    OP_CVTSI2SS_I32,
    OP_CVTSI2SS_I64,
    OP_CVTSI2SD_I32,
    OP_CVTSI2SD_I64,
    OP_CVTSS2SI32,
    OP_CVTTSS2SI32,
    OP_CVTSS2SI64,
    OP_CVTTSS2SI64,
    OP_CVTSD2SI32,
    OP_CVTTSD2SI32,
    OP_CVTSD2SI64,
    OP_CVTTSD2SI64,
    OP_COUNT,
    OP_FP32_FMA_FIRST = OP_FMADDSS,
    OP_FP32_FMA_LAST = OP_FNMSUBSS,
    OP_FP64_BINARY_FIRST = OP_ADDSD,
    OP_FP64_BINARY_LAST = OP_MAXSD,
    OP_FP64_FMA_FIRST = OP_FMADDSD,
    OP_FP64_FMA_LAST = OP_FNMSUBSD
} test_op;

static const char *opcode_name(test_op opcode) {
    switch (opcode) {
        case OP_ADDSS: return "addss";
        case OP_SUBSS: return "subss";
        case OP_MULSS: return "mulss";
        case OP_DIVSS: return "divss";
        case OP_MINSS: return "minss";
        case OP_MAXSS: return "maxss";
        case OP_SQRTSS: return "sqrtss";
        case OP_ROUNDSS: return "roundss";
        case OP_CMPSS: return "cmpss";
        case OP_FMADDSS: return "fmaddss";
        case OP_FMSUBSS: return "fmsubss";
        case OP_FNMADDSS: return "fnmaddss";
        case OP_FNMSUBSS: return "fnmsubss";
        case OP_COMISS: return "comiss";
        case OP_UCOMISS: return "ucomiss";
        case OP_ADDSD: return "addsd";
        case OP_SUBSD: return "subsd";
        case OP_MULSD: return "mulsd";
        case OP_DIVSD: return "divsd";
        case OP_MINSD: return "minsd";
        case OP_MAXSD: return "maxsd";
        case OP_SQRTSD: return "sqrtsd";
        case OP_ROUNDSD: return "roundsd";
        case OP_CMPSD: return "cmpsd";
        case OP_FMADDSD: return "fmaddsd";
        case OP_FMSUBSD: return "fmsubsd";
        case OP_FNMADDSD: return "fnmaddsd";
        case OP_FNMSUBSD: return "fnmsubsd";
        case OP_COMISD: return "comisd";
        case OP_UCOMISD: return "ucomisd";
        case OP_CVTSD2SS: return "cvtsd2ss";
        case OP_CVTSS2SD: return "cvtss2sd";
        case OP_VCVTPH2PS_LANE: return "vcvtph2ps_lane";
        case OP_VCVTPS2PH_LANE: return "vcvtps2ph_lane";
        case OP_CVTSI2SS_I32: return "cvtsi2ss_i32";
        case OP_CVTSI2SS_I64: return "cvtsi2ss_i64";
        case OP_CVTSI2SD_I32: return "cvtsi2sd_i32";
        case OP_CVTSI2SD_I64: return "cvtsi2sd_i64";
        case OP_CVTSS2SI32: return "cvtss2si32";
        case OP_CVTTSS2SI32: return "cvttss2si32";
        case OP_CVTSS2SI64: return "cvtss2si64";
        case OP_CVTTSS2SI64: return "cvttss2si64";
        case OP_CVTSD2SI32: return "cvtsd2si32";
        case OP_CVTTSD2SI32: return "cvttsd2si32";
        case OP_CVTSD2SI64: return "cvtsd2si64";
        case OP_CVTTSD2SI64: return "cvttsd2si64";
        case OP_COUNT:
        default: return "?";
    }
}

static bool parse_opcode(const char *name, test_op *p_opcode) {
    for (uint32_t opcode = 0; opcode < OP_COUNT; opcode++) {
        if (!strcmp(name, opcode_name((test_op)opcode))) {
            *p_opcode = (test_op)opcode;
            return true;
        }
    }
    return false;
}

static bool opcode_is_fp32_fma(test_op opcode) {
    return (opcode >= OP_FP32_FMA_FIRST) && (opcode <= OP_FP32_FMA_LAST);
}

static bool opcode_is_fp64_fma(test_op opcode) {
    return (opcode >= OP_FP64_FMA_FIRST) && (opcode <= OP_FP64_FMA_LAST);
}

static bool opcode_is_fp64_binary_op(test_op opcode) {
    return (opcode >= OP_FP64_BINARY_FIRST) && (opcode <= OP_FP64_BINARY_LAST);
}

static bool run_random_tests(uint64_t loops, uint64_t seed) {
    lehmer64_seed(seed);
    bool failed = false;
    uint32_t inner_loops = (loops > INNER_LOOPS) ? INNER_LOOPS : loops;
    for (uint64_t i = 0; i < loops; i += inner_loops) {
        uint64_t r1 = lehmer64();
        // All mask bits on (traps out of scope); exception-flag bits left fully random so both
        // clean and sticky-pre-set starting states are exercised. Rounding/FTZ/DAZ are also
        // random via r1&0xFFFF.
        uint32_t mxcsr = (r1 & 0xFFFF) | FPEMU_MXCSR_DEFAULT;
        // The enum order groups opcode families so range predicates above remain simple.
        test_op opcode = (test_op)((r1 >> 56) % OP_COUNT);
        for (uint32_t j = 0; j < inner_loops; j++) {
            uint64_t r0 = lehmer64();
            // FP64->FP32 cvt needs a second independent draw. Reusing r0 as the FP64 operand
            // means its high 32 bits are already being interpreted as an FP32 sign/exponent on
            // other opcodes, so entire regions of FP64 mantissa space are under-sampled.
            // Always draw r0b -- even for the FP32 opcodes -- so the stream advances uniformly
            // across opcodes.
            uint64_t r0b = lehmer64();
            uint64_t r0c = lehmer64();
            uint32_t a = r0 & 0xFFFFFFFF;
            uint32_t b = (uint32_t)(r0 >> 32);
            uint32_t c = r0b & 0xFFFFFFFF;
            uint64_t a64 = r0;
            uint64_t b64 = r0b;
            uint64_t c64 = r0c;
            uint32_t imm_round = (uint32_t)r0c & 15;
            uint32_t imm_cmp = (uint32_t)r0c & 31;
            uint32_t imm_f16 = (uint32_t)r0c & 7;
            uint64_t hw = 0;
            uint64_t sw = 0;
            uint32_t hw_mxcsr = mxcsr;
            uint32_t sw_mxcsr = mxcsr;
            switch (opcode) {
                case OP_ADDSS:
                    hw = hw_fp32_add(a, b, &hw_mxcsr);
                    sw = fpemu_fp32_add(a, b, &sw_mxcsr);
                    break;
                case OP_SUBSS:
                    hw = hw_fp32_sub(a, b, &hw_mxcsr);
                    sw = fpemu_fp32_sub(a, b, &sw_mxcsr);
                    break;
                case OP_MULSS:
                    hw = hw_fp32_mul(a, b, &hw_mxcsr);
                    sw = fpemu_fp32_mul(a, b, &sw_mxcsr);
                    break;
                case OP_DIVSS:
                    hw = hw_fp32_div(a, b, &hw_mxcsr);
                    sw = fpemu_fp32_div(a, b, &sw_mxcsr);
                    break;
                case OP_CVTSD2SS:
                    hw = hw_fp64_to_fp32(r0b, &hw_mxcsr);
                    sw = fpemu_fp64_to_fp32(r0b, &sw_mxcsr);
                    break;
                case OP_SQRTSS:
                    // Unary sqrt uses only operand `a`. `b` is drawn but ignored; keep the draw so
                    // the stream stays opcode-uniform.
                    hw = hw_fp32_sqrt(a, &hw_mxcsr);
                    sw = fpemu_fp32_sqrt(a, &sw_mxcsr);
                    break;
                case OP_CVTSS2SD:
                    hw = hw_fp32_to_fp64(a, &hw_mxcsr);
                    sw = fpemu_fp32_to_fp64(a, &sw_mxcsr);
                    break;
                case OP_CVTSI2SS_I32: {
                    // INT32 -> FP32. Operand is int32_t reinterpreted from the 32 bits of a.
                    int32_t ia = (int32_t)a;
                    hw = hw_int32_to_fp32(ia, &hw_mxcsr);
                    sw = fpemu_int32_to_fp32(ia, &sw_mxcsr);
                    break;
                }
                case OP_CVTSI2SS_I64: {
                    // INT64 -> FP32. Use r0 (already drawn) as the int64_t operand.
                    int64_t ia = (int64_t)r0;
                    hw = hw_int64_to_fp32(ia, &hw_mxcsr);
                    sw = fpemu_int64_to_fp32(ia, &sw_mxcsr);
                    break;
                }
                case OP_CVTSI2SD_I32: {
                    // INT32 -> FP64. Always exact; still runs full hw/sw compare to verify the
                    // software path never touches MXCSR and produces exact bits.
                    int32_t ia = (int32_t)a;
                    hw = hw_int32_to_fp64(ia, &hw_mxcsr);
                    sw = fpemu_int32_to_fp64(ia, &sw_mxcsr);
                    break;
                }
                case OP_CVTSI2SD_I64: {
                    // INT64 -> FP64. Rounding needed; #P is the only reachable flag.
                    int64_t ia = (int64_t)r0;
                    hw = hw_int64_to_fp64(ia, &hw_mxcsr);
                    sw = fpemu_int64_to_fp64(ia, &sw_mxcsr);
                    break;
                }
                case OP_CVTSS2SI32:
                    hw = hw_fp32_to_i32(a, &hw_mxcsr);
                    sw = fpemu_fp32_to_i32(a, &sw_mxcsr);
                    break;
                case OP_CVTTSS2SI32:
                    hw = hw_fp32_to_i32_trunc(a, &hw_mxcsr);
                    sw = fpemu_fp32_to_i32_trunc(a, &sw_mxcsr);
                    break;
                case OP_CVTSS2SI64:
                    hw = hw_fp32_to_i64(a, &hw_mxcsr);
                    sw = fpemu_fp32_to_i64(a, &sw_mxcsr);
                    break;
                case OP_CVTTSS2SI64:
                    hw = hw_fp32_to_i64_trunc(a, &hw_mxcsr);
                    sw = fpemu_fp32_to_i64_trunc(a, &sw_mxcsr);
                    break;
                case OP_CVTSD2SI32:
                    hw = hw_fp64_to_i32(r0b, &hw_mxcsr);
                    sw = fpemu_fp64_to_i32(r0b, &sw_mxcsr);
                    break;
                case OP_CVTTSD2SI32:
                    hw = hw_fp64_to_i32_trunc(r0b, &hw_mxcsr);
                    sw = fpemu_fp64_to_i32_trunc(r0b, &sw_mxcsr);
                    break;
                case OP_CVTSD2SI64:
                    hw = hw_fp64_to_i64(r0b, &hw_mxcsr);
                    sw = fpemu_fp64_to_i64(r0b, &sw_mxcsr);
                    break;
                case OP_CVTTSD2SI64:
                    hw = hw_fp64_to_i64_trunc(r0b, &hw_mxcsr);
                    sw = fpemu_fp64_to_i64_trunc(r0b, &sw_mxcsr);
                    break;
                case OP_ADDSD:
                    hw = hw_fp64_add(a64, b64, &hw_mxcsr);
                    sw = fpemu_fp64_add(a64, b64, &sw_mxcsr);
                    break;
                case OP_SUBSD:
                    hw = hw_fp64_sub(a64, b64, &hw_mxcsr);
                    sw = fpemu_fp64_sub(a64, b64, &sw_mxcsr);
                    break;
                case OP_MULSD:
                    hw = hw_fp64_mul(a64, b64, &hw_mxcsr);
                    sw = fpemu_fp64_mul(a64, b64, &sw_mxcsr);
                    break;
                case OP_DIVSD:
                    hw = hw_fp64_div(a64, b64, &hw_mxcsr);
                    sw = fpemu_fp64_div(a64, b64, &sw_mxcsr);
                    break;
                case OP_SQRTSD:
                    // Unary sqrt uses only operand a64; b64 is still drawn so opcode streams
                    // consume random inputs uniformly.
                    hw = hw_fp64_sqrt(a64, &hw_mxcsr);
                    sw = fpemu_fp64_sqrt(a64, &sw_mxcsr);
                    break;
                case OP_FMADDSS:
                    hw = hw_fp32_fmadd(a, b, c, &hw_mxcsr);
                    sw = fpemu_fp32_fmadd(a, b, c, &sw_mxcsr);
                    break;
                case OP_FMSUBSS:
                    hw = hw_fp32_fmsub(a, b, c, &hw_mxcsr);
                    sw = fpemu_fp32_fmsub(a, b, c, &sw_mxcsr);
                    break;
                case OP_FNMADDSS:
                    hw = hw_fp32_fnmadd(a, b, c, &hw_mxcsr);
                    sw = fpemu_fp32_fnmadd(a, b, c, &sw_mxcsr);
                    break;
                case OP_FNMSUBSS:
                    hw = hw_fp32_fnmsub(a, b, c, &hw_mxcsr);
                    sw = fpemu_fp32_fnmsub(a, b, c, &sw_mxcsr);
                    break;
                case OP_FMADDSD:
                    hw = hw_fp64_fmadd(a64, b64, c64, &hw_mxcsr);
                    sw = fpemu_fp64_fmadd(a64, b64, c64, &sw_mxcsr);
                    break;
                case OP_FMSUBSD:
                    hw = hw_fp64_fmsub(a64, b64, c64, &hw_mxcsr);
                    sw = fpemu_fp64_fmsub(a64, b64, c64, &sw_mxcsr);
                    break;
                case OP_FNMADDSD:
                    hw = hw_fp64_fnmadd(a64, b64, c64, &hw_mxcsr);
                    sw = fpemu_fp64_fnmadd(a64, b64, c64, &sw_mxcsr);
                    break;
                case OP_FNMSUBSD:
                    hw = hw_fp64_fnmsub(a64, b64, c64, &hw_mxcsr);
                    sw = fpemu_fp64_fnmsub(a64, b64, c64, &sw_mxcsr);
                    break;
                case OP_MINSS:
                    hw = hw_fp32_min(a, b, &hw_mxcsr);
                    sw = fpemu_fp32_min(a, b, &sw_mxcsr);
                    break;
                case OP_MAXSS:
                    hw = hw_fp32_max(a, b, &hw_mxcsr);
                    sw = fpemu_fp32_max(a, b, &sw_mxcsr);
                    break;
                case OP_MINSD:
                    hw = hw_fp64_min(a64, b64, &hw_mxcsr);
                    sw = fpemu_fp64_min(a64, b64, &sw_mxcsr);
                    break;
                case OP_MAXSD:
                    hw = hw_fp64_max(a64, b64, &hw_mxcsr);
                    sw = fpemu_fp64_max(a64, b64, &sw_mxcsr);
                    break;
                case OP_ROUNDSS:
                    hw = hw_fp32_round(a, imm_round, &hw_mxcsr);
                    sw = fpemu_fp32_round(a, imm_round, &sw_mxcsr);
                    break;
                case OP_ROUNDSD:
                    hw = hw_fp64_round(a64, imm_round, &hw_mxcsr);
                    sw = fpemu_fp64_round(a64, imm_round, &sw_mxcsr);
                    break;
                case OP_CMPSS:
                    hw = hw_fp32_cmp(a, b, imm_cmp, &hw_mxcsr);
                    sw = fpemu_fp32_cmp(a, b, imm_cmp, &sw_mxcsr);
                    break;
                case OP_CMPSD:
                    hw = hw_fp64_cmp(a64, b64, imm_cmp, &hw_mxcsr);
                    sw = fpemu_fp64_cmp(a64, b64, imm_cmp, &sw_mxcsr);
                    break;
                case OP_VCVTPH2PS_LANE:
                    hw = hw_fp16_to_fp32((uint16_t)a, &hw_mxcsr);
                    sw = fpemu_fp16_to_fp32((uint16_t)a, &sw_mxcsr);
                    break;
                case OP_VCVTPS2PH_LANE:
                    hw = hw_fp32_to_fp16(a, imm_f16, &hw_mxcsr);
                    sw = fpemu_fp32_to_fp16(a, imm_f16, &sw_mxcsr);
                    break;
                case OP_COMISS:
                    hw = hw_fp32_comi(a, b, &hw_mxcsr);
                    sw = fpemu_fp32_comi(a, b, &sw_mxcsr);
                    break;
                case OP_UCOMISS:
                    hw = hw_fp32_ucomi(a, b, &hw_mxcsr);
                    sw = fpemu_fp32_ucomi(a, b, &sw_mxcsr);
                    break;
                case OP_COMISD:
                    hw = hw_fp64_comi(a64, b64, &hw_mxcsr);
                    sw = fpemu_fp64_comi(a64, b64, &sw_mxcsr);
                    break;
                case OP_UCOMISD:
                    hw = hw_fp64_ucomi(a64, b64, &hw_mxcsr);
                    sw = fpemu_fp64_ucomi(a64, b64, &sw_mxcsr);
                    break;
                default:
                    TEST_ASSERT(false, "unexpected opcode=%u", (uint32_t)opcode);
            }
            if ((hw != sw) || (hw_mxcsr != sw_mxcsr)) {
                failed = true;
                if (opcode == OP_CVTSD2SS) {
                    printf("%llu: %s 0x%016llx (mxcsr=0x%x) -> "
                           "0x%x/0x%x (hw), 0x%x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (unsigned long long)r0b, mxcsr, (uint32_t)hw, hw_mxcsr,
                           (uint32_t)sw, sw_mxcsr);
                } else if (opcode == OP_VCVTPH2PS_LANE) {
                    printf("%llu: %s 0x%x (mxcsr=0x%x) -> 0x%x/0x%x (hw), 0x%x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           a & 0xFFFFu, mxcsr, (uint32_t)hw, hw_mxcsr,
                           (uint32_t)sw, sw_mxcsr);
                } else if (opcode == OP_VCVTPS2PH_LANE) {
                    printf("%llu: %s 0x%x imm=0x%x (mxcsr=0x%x) -> "
                           "0x%x/0x%x (hw), 0x%x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), a,
                           imm_f16, mxcsr, (uint32_t)hw, hw_mxcsr,
                           (uint32_t)sw, sw_mxcsr);
                } else if (opcode == OP_CVTSS2SD) {
                    printf("%llu: %s 0x%x (mxcsr=0x%x) -> "
                           "0x%016llx/0x%x (hw), 0x%016llx/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), a, mxcsr,
                           (unsigned long long)hw, hw_mxcsr,
                           (unsigned long long)sw, sw_mxcsr);
                } else if (opcode == OP_CVTSI2SS_I32) {
                    printf("%llu: %s %d (mxcsr=0x%x) -> "
                           "0x%x/0x%x (hw), 0x%x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), (int32_t)a,
                           mxcsr, (uint32_t)hw, hw_mxcsr, (uint32_t)sw, sw_mxcsr);
                } else if (opcode == OP_CVTSI2SS_I64) {
                    printf("%llu: %s %lld (mxcsr=0x%x) -> "
                           "0x%x/0x%x (hw), 0x%x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (long long)(int64_t)r0, mxcsr, (uint32_t)hw, hw_mxcsr,
                           (uint32_t)sw, sw_mxcsr);
                } else if (opcode == OP_CVTSI2SD_I32) {
                    printf("%llu: %s %d (mxcsr=0x%x) -> "
                           "0x%016llx/0x%x (hw), 0x%016llx/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), (int32_t)a,
                           mxcsr, (unsigned long long)hw, hw_mxcsr,
                           (unsigned long long)sw, sw_mxcsr);
                } else if (opcode == OP_CVTSI2SD_I64) {
                    printf("%llu: %s %lld (mxcsr=0x%x) -> "
                           "0x%016llx/0x%x (hw), 0x%016llx/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (long long)(int64_t)r0, mxcsr, (unsigned long long)hw,
                           hw_mxcsr, (unsigned long long)sw, sw_mxcsr);
                } else if ((opcode == OP_CVTSS2SI32) || (opcode == OP_CVTTSS2SI32)) {
                    printf("%llu: %s 0x%x (mxcsr=0x%x) -> "
                           "0x%08x/0x%x (hw), 0x%08x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), a, mxcsr,
                           (uint32_t)hw, hw_mxcsr, (uint32_t)sw, sw_mxcsr);
                } else if ((opcode == OP_CVTSS2SI64) || (opcode == OP_CVTTSS2SI64)) {
                    printf("%llu: %s 0x%x (mxcsr=0x%x) -> "
                           "0x%016llx/0x%x (hw), 0x%016llx/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), a, mxcsr,
                           (unsigned long long)hw, hw_mxcsr,
                           (unsigned long long)sw, sw_mxcsr);
                } else if ((opcode == OP_CVTSD2SI32) || (opcode == OP_CVTTSD2SI32)) {
                    printf("%llu: %s 0x%016llx (mxcsr=0x%x) -> "
                           "0x%08x/0x%x (hw), 0x%08x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (unsigned long long)r0b, mxcsr, (uint32_t)hw, hw_mxcsr,
                           (uint32_t)sw, sw_mxcsr);
                } else if ((opcode == OP_CVTSD2SI64) || (opcode == OP_CVTTSD2SI64)) {
                    printf("%llu: %s 0x%016llx (mxcsr=0x%x) -> "
                           "0x%016llx/0x%x (hw), 0x%016llx/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (unsigned long long)r0b, mxcsr, (unsigned long long)hw,
                           hw_mxcsr, (unsigned long long)sw, sw_mxcsr);
                } else if (opcode == OP_ROUNDSD) {
                    printf("%llu: %s 0x%016llx imm=0x%x (mxcsr=0x%x) -> "
                           "0x%016llx/0x%x (hw), 0x%016llx/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (unsigned long long)a64, imm_round, mxcsr,
                           (unsigned long long)hw, hw_mxcsr,
                           (unsigned long long)sw, sw_mxcsr);
                } else if (opcode == OP_CMPSD) {
                    printf("%llu: %s 0x%016llx 0x%016llx imm=0x%x (mxcsr=0x%x) -> "
                           "0x%016llx/0x%x (hw), 0x%016llx/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (unsigned long long)a64, (unsigned long long)b64, imm_cmp, mxcsr,
                           (unsigned long long)hw, hw_mxcsr,
                           (unsigned long long)sw, sw_mxcsr);
                } else if (opcode == OP_ROUNDSS) {
                    printf("%llu: %s 0x%x imm=0x%x (mxcsr=0x%x) -> "
                           "0x%x/0x%x (hw), 0x%x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), a, imm_round,
                           mxcsr, (uint32_t)hw, hw_mxcsr, (uint32_t)sw, sw_mxcsr);
                } else if (opcode == OP_CMPSS) {
                    printf("%llu: %s 0x%x 0x%x imm=0x%x (mxcsr=0x%x) -> "
                           "0x%x/0x%x (hw), 0x%x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), a, b, imm_cmp,
                           mxcsr, (uint32_t)hw, hw_mxcsr, (uint32_t)sw, sw_mxcsr);
                } else if ((opcode == OP_COMISD) || (opcode == OP_UCOMISD)) {
                    printf("%llu: %s 0x%016llx 0x%016llx (mxcsr=0x%x) -> "
                           "hw_flags=0x%x/0x%x sw_flags=0x%x/0x%x\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (unsigned long long)a64, (unsigned long long)b64, mxcsr,
                           (uint32_t)hw, hw_mxcsr, (uint32_t)sw, sw_mxcsr);
                } else if ((opcode == OP_COMISS) || (opcode == OP_UCOMISS)) {
                    printf("%llu: %s 0x%x 0x%x (mxcsr=0x%x) -> "
                           "hw_flags=0x%x/0x%x sw_flags=0x%x/0x%x\n",
                           (unsigned long long)(i + j), opcode_name(opcode), a, b,
                           mxcsr, (uint32_t)hw, hw_mxcsr, (uint32_t)sw, sw_mxcsr);
                } else if (opcode_is_fp64_fma(opcode)) {
                    printf("%llu: %s 0x%016llx 0x%016llx 0x%016llx (mxcsr=0x%x) "
                           "-> 0x%016llx/0x%x (hw), 0x%016llx/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (unsigned long long)a64, (unsigned long long)b64,
                           (unsigned long long)c64, mxcsr, (unsigned long long)hw,
                           hw_mxcsr, (unsigned long long)sw, sw_mxcsr);
                } else if (opcode == OP_SQRTSD) {
                    printf("%llu: %s 0x%016llx (mxcsr=0x%x) -> "
                           "0x%016llx/0x%x (hw), 0x%016llx/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (unsigned long long)a64, mxcsr, (unsigned long long)hw,
                           hw_mxcsr, (unsigned long long)sw, sw_mxcsr);
                } else if (opcode_is_fp64_binary_op(opcode)) {
                    printf("%llu: %s 0x%016llx 0x%016llx (mxcsr=0x%x) -> "
                           "0x%016llx/0x%x (hw), 0x%016llx/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode),
                           (unsigned long long)a64, (unsigned long long)b64, mxcsr,
                           (unsigned long long)hw, hw_mxcsr, (unsigned long long)sw, sw_mxcsr);
                } else if (opcode_is_fp32_fma(opcode)) {
                    printf("%llu: %s 0x%x 0x%x 0x%x (mxcsr=0x%x) -> "
                           "0x%x/0x%x (hw), 0x%x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), a, b,
                           c, mxcsr, (uint32_t)hw, hw_mxcsr, (uint32_t)sw, sw_mxcsr);
                } else if (opcode == OP_SQRTSS) {
                    printf("%llu: %s 0x%x (mxcsr=0x%x) -> "
                           "0x%x/0x%x (hw), 0x%x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), a, mxcsr,
                           (uint32_t)hw, hw_mxcsr, (uint32_t)sw, sw_mxcsr);
                } else {
                    printf("%llu: %s 0x%x 0x%x (mxcsr=0x%x) -> 0x%x/0x%x (hw), "
                           "0x%x/0x%x (sw)\n",
                           (unsigned long long)(i + j), opcode_name(opcode), a, b,
                           mxcsr, (uint32_t)hw, hw_mxcsr, (uint32_t)sw, sw_mxcsr);
                }
            }
        }
    }
    if (!s_quiet_success || failed) {
        printf("%s ran %llu cases\n", failed ? "FAILED" : "DONE", (unsigned long long)loops);
    }
    return !failed;
}

static bool opcode_supports_exhaustive(test_op opcode) {
    switch (opcode) {
        case OP_SQRTSS:
        case OP_ROUNDSS:
        case OP_CVTSS2SD:
        case OP_VCVTPH2PS_LANE:
        case OP_VCVTPS2PH_LANE:
        case OP_CVTSI2SS_I32:
        case OP_CVTSI2SD_I32:
        case OP_CVTSS2SI32:
        case OP_CVTTSS2SI32:
        case OP_CVTSS2SI64:
        case OP_CVTTSS2SI64:
            return true;
        default:
            return false;
    }
}

static bool parse_exhaustive_op(const char *name, test_op *p_opcode) {
    test_op opcode;
    if (!parse_opcode(name, &opcode) || !opcode_supports_exhaustive(opcode)) {
        return false;
    }
    *p_opcode = opcode;
    return true;
}

static uint32_t exhaustive_op_input_bits(test_op opcode) {
    return (opcode == OP_VCVTPH2PS_LANE) ? 16 : 32;
}

static void exhaustive_eval(test_op opcode, uint64_t input,
                            uint32_t init_mxcsr, uint32_t imm, uint64_t *p_hw,
                            uint32_t *p_hw_mxcsr, uint64_t *p_sw, uint32_t *p_sw_mxcsr) {
    uint32_t a32 = (uint32_t)input;
    uint16_t a16 = (uint16_t)input;
    int32_t i32 = (int32_t)a32;

    *p_hw_mxcsr = init_mxcsr;
    *p_sw_mxcsr = init_mxcsr;
    switch (opcode) {
        case OP_SQRTSS:
            *p_hw = hw_fp32_sqrt(a32, p_hw_mxcsr);
            *p_sw = fpemu_fp32_sqrt(a32, p_sw_mxcsr);
            break;
        case OP_ROUNDSS:
            *p_hw = hw_fp32_round(a32, imm, p_hw_mxcsr);
            *p_sw = fpemu_fp32_round(a32, imm, p_sw_mxcsr);
            break;
        case OP_CVTSS2SD:
            *p_hw = hw_fp32_to_fp64(a32, p_hw_mxcsr);
            *p_sw = fpemu_fp32_to_fp64(a32, p_sw_mxcsr);
            break;
        case OP_VCVTPH2PS_LANE:
            *p_hw = hw_fp16_to_fp32(a16, p_hw_mxcsr);
            *p_sw = fpemu_fp16_to_fp32(a16, p_sw_mxcsr);
            break;
        case OP_VCVTPS2PH_LANE:
            *p_hw = hw_fp32_to_fp16(a32, imm, p_hw_mxcsr);
            *p_sw = fpemu_fp32_to_fp16(a32, imm, p_sw_mxcsr);
            break;
        case OP_CVTSI2SS_I32:
            *p_hw = hw_int32_to_fp32(i32, p_hw_mxcsr);
            *p_sw = fpemu_int32_to_fp32(i32, p_sw_mxcsr);
            break;
        case OP_CVTSI2SD_I32:
            *p_hw = hw_int32_to_fp64(i32, p_hw_mxcsr);
            *p_sw = fpemu_int32_to_fp64(i32, p_sw_mxcsr);
            break;
        case OP_CVTSS2SI32:
            *p_hw = hw_fp32_to_i32(a32, p_hw_mxcsr);
            *p_sw = fpemu_fp32_to_i32(a32, p_sw_mxcsr);
            break;
        case OP_CVTTSS2SI32:
            *p_hw = hw_fp32_to_i32_trunc(a32, p_hw_mxcsr);
            *p_sw = fpemu_fp32_to_i32_trunc(a32, p_sw_mxcsr);
            break;
        case OP_CVTSS2SI64:
            *p_hw = hw_fp32_to_i64(a32, p_hw_mxcsr);
            *p_sw = fpemu_fp32_to_i64(a32, p_sw_mxcsr);
            break;
        case OP_CVTTSS2SI64:
            *p_hw = hw_fp32_to_i64_trunc(a32, p_hw_mxcsr);
            *p_sw = fpemu_fp32_to_i64_trunc(a32, p_sw_mxcsr);
            break;
        default:
            TEST_ASSERT(false, "unexpected exhaustive opcode=%u", (uint32_t)opcode);
    }
}

static bool run_exhaustive(test_op opcode, uint32_t rm, bool daz, bool ftz,
                           uint32_t imm, uint32_t shard_index,
                           uint32_t shard_count) {
    TEST_ASSERT(rm < 4, "invalid rm=%u", rm);
    TEST_ASSERT(opcode_supports_exhaustive(opcode),
                "opcode does not support exhaustive testing: %s", opcode_name(opcode));
    TEST_ASSERT(shard_count >= 1, "invalid shard_count=%u", shard_count);
    TEST_ASSERT(shard_index < shard_count, "invalid shard_index=%u shard_count=%u",
                shard_index, shard_count);

    uint32_t input_bits = exhaustive_op_input_bits(opcode);
    const uint64_t total_inputs = 1ull << input_bits;
    const uint32_t init_mxcsr =
        FPEMU_MXCSR_DEFAULT | (rm << 13) | (daz ? FPEMU_MXCSR_DAZ : 0) |
        (ftz ? FPEMU_MXCSR_FTZ : 0);
    const uint64_t full_shard_inputs =
        (shard_index < total_inputs) ? ((total_inputs - 1 - shard_index) /
                                        shard_count) + 1
                                     : 0;
    bool failed = false;
    uint64_t ran = 0;
    uint64_t printed = 0;

    for (uint64_t input = shard_index; input < total_inputs;
         input += shard_count) {
        uint64_t hw, sw;
        uint32_t hw_mxcsr, sw_mxcsr;
        exhaustive_eval(opcode, input, init_mxcsr, imm, &hw, &hw_mxcsr, &sw, &sw_mxcsr);
        if ((hw != sw) || (hw_mxcsr != sw_mxcsr)) {
            failed = true;
            if (printed < MAX_PRINT) {
                printf("%s rm=%u daz=%u ftz=%u imm=%u input=0x%0*llx -> "
                       "0x%016llx/0x%04x (hw), 0x%016llx/0x%04x (sw)\n",
                       opcode_name(opcode), rm, daz ? 1 : 0, ftz ? 1 : 0,
                       imm, input_bits / 4, (unsigned long long)input,
                       (unsigned long long)hw, hw_mxcsr, (unsigned long long)sw, sw_mxcsr);
                printed++;
            }
        }
        ran++;
    }

    if (!s_quiet_success || failed) {
        printf("%s exhaustive %s rm=%u daz=%u ftz=%u imm=%u shard %llu/%llu "
               "ran %llu/%llu cases\n",
               failed ? "FAILED" : "DONE", opcode_name(opcode), rm, daz ? 1 : 0,
               ftz ? 1 : 0, imm, (unsigned long long)shard_index,
               (unsigned long long)shard_count, (unsigned long long)ran,
               (unsigned long long)full_shard_inputs);
    }
    return !failed;
}

// ------------------------------------------------------------------------------------
// Directed edge-case suite.
//
// The randomized harness pre-sets DE/UE/PE in MXCSR for each outer block, which masks any
// software path that fails to raise those flags. The suite below runs a cross product of curated
// FP32/FP64 encodings x 4 rounding modes x {FTZ off,on} x {DAZ off,on} starting from a clean
// MXCSR and compares both the result bits and the resulting MXCSR against the hardware helpers.
// ------------------------------------------------------------------------------------

typedef struct fp32_case {
    uint32_t bits;
    const char *name;
} fp32_case;
static const fp32_case s_fp32_cases[] = {
    {0x00000000, "+0"},
    {0x80000000, "-0"},
    {0x00000001, "+min_sub"},
    {0x80000001, "-min_sub"},
    {0x007FFFFF, "+max_sub"},
    {0x807FFFFF, "-max_sub"},
    {0x00000002, "+sub2"},
    {0x00400000, "+mid_sub"},
    {0x00800000, "+min_nor"},
    {0x80800000, "-min_nor"},
    {0x00800001, "+min_nor+1ulp"},
    {0x3F800000, "+1.0"},
    {0xBF800000, "-1.0"},
    {0x3F800001, "+1+1ulp"},
    {0x3F800002, "+1+2ulp"},
    {0x3F7FFFFF, "+1-0.5ulp"},
    {0x40000000, "+2.0"},
    {0x3F000000, "+0.5"},
    {0x40400000, "+3.0"},
    {0x40A00000, "+5.0"},
    {0x3F400000, "+0.75"},
    {0x40490FDB, "+pi"},
    {0x7F000000, "+2^127"},
    // 2^103 paired with +max_fin is the classic phase-1 carry-into-overflow probe: mant_result
    // comes out with mantN=0x7FFFFF, guardN=1, which rounds up to 0x800000 and increments expN
    // from 254 to 255 -- exercising the overflow path *via* the carry rather than directly.
    // Verified on hardware: RN -> +inf with O|P, RZ/RD -> max_fin with O|P. Without this case the
    // carry branch in round_to_fp32 is not demonstrably exercised by the directed suite.
    {0x73000000, "+2^103"},
    {0x7F7FFFFF, "+max_fin"},
    {0xFF7FFFFF, "-max_fin"},
    {0x7F7FFFFE, "+max_fin-1ulp"},
    {0x7F7FFFFC, "+max_fin-3ulp"},
    // Overflow-adjacent: these multiplied by small factors let round_to_fp32's phase-1 carry
    // into overflow actually trigger.
    {0x3FFFFFFF, "+2-1ulp"},
    {0x3FFFFFFE, "+2-2ulp"},
    {0x40000001, "+2+1ulp"},
    // Subnormal boundary spread for directed-rounding behavior.
    {0x00400001, "+mid_sub+1ulp"},
    {0x003FFFFF, "+mid_sub-1ulp"},
    {0x007FFFFE, "+max_sub-1ulp"},
    {0x00800002, "+min_nor+2ulp"},
    {0x007FFFFD, "+max_sub-2ulp"},
    // FP16C FP32->FP16 underflow boundary around half max-subnormal, midpoint, and min-normal.
    {0x387FC000, "+hmax_sub"},
    {0xB87FC000, "-hmax_sub"},
    {0x387FC001, "+hmax_sub+1ulp"},
    {0xB87FC001, "-hmax_sub-1ulp"},
    {0x387FE000, "+hsub_hnormal_mid"},
    {0xB87FE000, "-hsub_hnormal_mid"},
    {0x387FE001, "+hsub_hnormal_mid+1ulp"},
    {0xB87FE001, "-hsub_hnormal_mid-1ulp"},
    {0x387FEFFF, "+hsub_hnormal_3q-1ulp"},
    {0xB87FEFFF, "-hsub_hnormal_3q-1ulp"},
    {0x387FF000, "+hsub_hnormal_3q"},
    {0xB87FF000, "-hsub_hnormal_3q"},
    {0x38800000, "+hmin_nor"},
    {0xB8800000, "-hmin_nor"},
    // Half-ULP "exact halfway" neighborhoods for RNE tie-to-even probes in add/sub:
    {0x3F800003, "+1+3ulp"},
    {0x3F800004, "+1+4ulp"},
    // A finite value whose product with (1+1ulp) tips into overflow under some rounding modes.
    {0x7F000001, "+2^127+1ulp"},
    {0x7F7FFFFB, "+max_fin-4ulp"},
    {0x7F800000, "+inf"},
    {0xFF800000, "-inf"},
    {0x7FC00000, "+qnan"},
    {0xFFC00000, "-qnan"},
    {0x7FC12345, "+qnan_payload"},
    {0x7F800001, "+snan"},
    {0xFF800001, "-snan"},
    {0x7FBFFFFF, "+snan_max"},
};
static const size_t s_fp32_cases_n =
    sizeof(s_fp32_cases) / sizeof(s_fp32_cases[0]);

typedef struct fp16_case {
    uint16_t bits;
    const char *name;
} fp16_case;
static const fp16_case s_fp16_cases[] = {
    {0x0000, "+0h"},
    {0x8000, "-0h"},
    {0x0001, "+min_subh"},
    {0x8001, "-min_subh"},
    {0x0002, "+sub2h"},
    {0x0200, "+mid_subh"},
    {0x03FF, "+max_subh"},
    {0x83FF, "-max_subh"},
    {0x0400, "+min_norh"},
    {0x8400, "-min_norh"},
    {0x0401, "+min_norh+1ulp"},
    {0x3C00, "+1.0h"},
    {0xBC00, "-1.0h"},
    {0x3C01, "+1h+1ulp"},
    {0x3BFF, "+1h-1ulp"},
    {0x4000, "+2.0h"},
    {0x3800, "+0.5h"},
    {0x4200, "+3.0h"},
    {0x7BFF, "+max_finh"},
    {0xFBFF, "-max_finh"},
    {0x7C00, "+infh"},
    {0xFC00, "-infh"},
    {0x7E00, "+qnanh"},
    {0xFE00, "-qnanh"},
    {0x7E55, "+qnan_payloadh"},
    {0x7C01, "+snanh"},
    {0xFC01, "-snanh"},
    {0x7DFF, "+snan_maxh"},
};
static const size_t s_fp16_cases_n =
    sizeof(s_fp16_cases) / sizeof(s_fp16_cases[0]);

typedef struct fp64_case {
    uint64_t bits;
    const char *name;
} fp64_case;
static const fp64_case s_fp64_cases[] = {
    {0x0000000000000000ull, "+0"},
    {0x8000000000000000ull, "-0"},
    {0x0000000000000001ull, "+min_sub64"},
    {0x8000000000000001ull, "-min_sub64"},
    {0x0000000000000002ull, "+sub2_64"},
    {0x000FFFFFFFFFFFFFull, "+max_sub64"},
    {0x800FFFFFFFFFFFFFull, "-max_sub64"},
    {0x0010000000000000ull, "+min_nor64"},
    {0x8010000000000000ull, "-min_nor64"},
    {0x0010000000000001ull, "+min_nor64+1ulp"},
    {0x3FF0000000000000ull, "+1.0"},
    {0xBFF0000000000000ull, "-1.0"},
    {0x3FE0000000000000ull, "+0.5"},
    {0x4000000000000000ull, "+2.0"},
    {0x3FF0000000000001ull, "+1+1ulp64"},
    {0x3FEFFFFFFFFFFFFFull, "+1-0.5ulp64"},
    {0x3FF0000000000002ull, "+1+2ulp64"},
    {0x7FEFFFFFFFFFFFFFull, "+max_fin64"},
    {0xFFEFFFFFFFFFFFFFull, "-max_fin64"},
    {0x7FEFFFFFFFFFFFFEull, "+max_fin64-1ulp"},
    {0x7FF0000000000000ull, "+inf"},
    {0xFFF0000000000000ull, "-inf"},
    {0x7FF8000000000000ull, "+qnan"},
    {0xFFF8000000000000ull, "-qnan"},
    {0x7FF0000000000001ull, "+snan"},
    {0xFFF0000000000001ull, "-snan"},
    {0x7FFFFFFFFFFFFFFFull, "+snan_max"},
    {0x3810000000000000ull, "fp32_min_nor"}, // 2^-126
    {0x38100000FFFFFFFFull, "fp32_min_nor+inexact"},
    {0x36A0000000000000ull, "fp32_min_sub"},   // 2^-149
    {0x3690000000000000ull, "fp32_min_sub/2"}, // rounds to 0 or min_sub
    {0x369FFFFFFFFFFFFFull, "halfway_min_sub"},
    {0x47EFFFFFE0000000ull, "fp32_max_fin"},
    {0x47EFFFFFE0000001ull, "fp32_max_fin+eps"},
    {0x47EFFFFFEFFFFFFFull, "just below halfway_to_overflow"},
    {0x47EFFFFFF0000000ull, "exactly halfway_to_overflow"},
    {0x47EFFFFFF0000001ull, "just above halfway_to_overflow"},
    {0x47F0000000000000ull, "above fp32_max"},
    // sNaN payloads that stress the payload-truncation path in fpemu_fp64_to_fp32.
    {0x7FF0000000000002ull, "snan_payload_2"},
    {0x7FF7FFFFFFFFFFFFull, "snan_max_payload"},
    {0x7FF2468ACE135791ull, "snan_mixed_payload"},
    {0xFFF2468ACE135791ull, "-snan_mixed_payload"},
    {0x7FF0000020000000ull, "snan_bit29_low_fp32_bit"},
    {0x7FF000001FFFFFFFull, "snan_low_bits_only_lost"},
    {0x7FF0001000000000ull, "snan_bit40_payload"},
    // qNaN payload variants (bit 51 set, additional payload).
    {0x7FFAAAAAAAAAAAAAull, "qnan_pattern_a"},
    {0xFFFAAAAAAAAAAAAAull, "-qnan_pattern_a"},
    {0x7FF8000000000001ull, "qnan_low_1_lost"},
    {0x7FF8000010000000ull, "qnan_bit28_lost"},
    {0x7FF8000020000000ull, "qnan_bit29_kept"},
    // FP64 mantissa patterns that exercise rounding-at-FP32-precision decisions. These are all
    // values representable exactly as FP32 or one ULP off -- good for phase-1/2 inexactness checks.
    {0x3FF0000020000000ull, "fp64 1.0 + fp32 1ulp"},
    {0x3FF0000010000000ull, "fp64 1.0 + 0.5 fp32 ulp (RNE tie down)"},
    {0x3FF0000030000000ull, "fp64 1.0 + 1.5 fp32 ulp (RNE tie up)"},
    {0x3FF0000018000000ull, "fp64 1.0 + 0.75 fp32 ulp"},
    // Near-max-fin rounding to trigger phase-1 carry to overflow.
    {0x47EFFFFFE8000000ull, "max_fin + 0.5 fp32 ulp (tie even down)"},
    {0x47EFFFFFD8000000ull, "max_fin-ulp + 0.75 ulp"},
    // Subnormal boundaries under directed rounding.
    {0x369FFFFFF0000000ull, "slightly < halfway_min_sub"},
    {0x36A0000010000000ull, "slightly > min_sub"},
    {0x36A8000000000000ull, "mid between min_sub and 2x min_sub"},
    {0x380FFFFFFFFFFFFFull, "just below min_nor (max sub fp32 in fp64)"},
    // Negative variants to exercise rm=1 (RU) vs rm=2 (RD) sign asymmetry.
    {0xBFF0000010000000ull, "-(1.0 + 0.5 fp32 ulp)"},
    {0xC7EFFFFFE8000000ull, "-(max_fin + 0.5 ulp)"},
    {0xB690000000000000ull, "-(fp32 min_sub/2)"},
    {0xB6A0000000000000ull, "-fp32_min_sub"},
};
static const size_t s_fp64_cases_n =
    sizeof(s_fp64_cases) / sizeof(s_fp64_cases[0]);

typedef struct fp32_to_i_case {
    uint32_t bits;
    const char *name;
} fp32_to_i_case;
static const fp32_to_i_case s_fp32_to_i_cases[] = {
    {0x00000000, "+0"},
    {0x80000000, "-0"},
    {0x00000001, "+min_sub"},
    {0x80000001, "-min_sub"},
    {0x007FFFFF, "+max_sub"},
    {0x807FFFFF, "-max_sub"},
    {0x3EFFFFFF, "+0.49999997"},
    {0xBEFFFFFF, "-0.49999997"},
    {0x3F000000, "+0.5"},
    {0xBF000000, "-0.5"},
    {0x3F7FFFFF, "+0.99999994"},
    {0xBF7FFFFF, "-0.99999994"},
    {0x3F800000, "+1.0"},
    {0xBF800000, "-1.0"},
    {0x3FC00000, "+1.5"},
    {0xBFC00000, "-1.5"},
    {0xBF400000, "-0.75"},
    {0xC0000000, "-2.0"},
    {0x4B7FFFFF, "+2^24-1"},
    {0x4B800000, "+2^24"},
    {0x4B800001, "+2^24+2"},
    {0x4EFFFFFF, "+(2^31-128)"},
    {0x4F000000, "+2^31"},
    {0xCEFFFFFF, "-(2^31-128)"},
    {0xCF000000, "-2^31"},
    {0x5EFFFFFF, "+(2^63-2^39)"},
    {0x5F000000, "+2^63"},
    {0xDEFFFFFF, "-(2^63-2^39)"},
    {0xDF000000, "-2^63"},
    {0x7F800000, "+inf"},
    {0xFF800000, "-inf"},
    {0x7FC00000, "+qnan"},
    {0x7F800001, "+snan"},
};
static const size_t s_fp32_to_i_cases_n =
    sizeof(s_fp32_to_i_cases) / sizeof(s_fp32_to_i_cases[0]);

typedef struct fp64_to_i_case {
    uint64_t bits;
    const char *name;
} fp64_to_i_case;
static const fp64_to_i_case s_fp64_to_i_cases[] = {
    {0x0000000000000000ull, "+0"},
    {0x8000000000000000ull, "-0"},
    {0x0000000000000001ull, "+min_sub64"},
    {0x8000000000000001ull, "-min_sub64"},
    {0x000FFFFFFFFFFFFFull, "+max_sub64"},
    {0x800FFFFFFFFFFFFFull, "-max_sub64"},
    {0x3FDFFFFFFFFFFFFFull, "+0.49999999999999994"},
    {0xBFDFFFFFFFFFFFFFull, "-0.49999999999999994"},
    {0x3FE0000000000000ull, "+0.5"},
    {0xBFE0000000000000ull, "-0.5"},
    {0x3FEFFFFFFFFFFFFFull, "+0.9999999999999999"},
    {0xBFEFFFFFFFFFFFFFull, "-0.9999999999999999"},
    {0x3FF0000000000000ull, "+1.0"},
    {0x3FF8000000000000ull, "+1.5"},
    {0xBFF8000000000000ull, "-1.5"},
    {0x4008000000000000ull, "+3.0"},
    {0x41DFFFFFFFC00000ull, "+(2^31-1)"},
    {0xC1DFFFFFFFC00000ull, "-(2^31-1)"},
    {0x41E0000000000000ull, "+2^31"},
    {0x41DFFFFFFFE00000ull, "+(2^31-0.5)"},
    {0x41E0000000100000ull, "+(2^31+0.5)"},
    {0xC1E0000000000000ull, "-2^31"},
    {0xC1DFFFFFFFE00000ull, "-(2^31-0.5)"},
    {0x43E0000000000000ull, "+2^63"},
    {0x43DFFFFFFFFFFFFFull, "+(2^63-1024)"},
    {0x43E0000000000001ull, "+(2^63+2048)"},
    {0x43E0000000000002ull, "+(2^63+4096)"},
    {0xC3E0000000000000ull, "-2^63"},
    {0xC3DFFFFFFFFFFFFFull, "-(2^63-1024)"},
    {0xC3E0000000000001ull, "-(2^63-2048)"},
    {0x7FF0000000000000ull, "+inf"},
    {0xFFF0000000000000ull, "-inf"},
    {0x7FF0000000000001ull, "+snan"},
    {0xFFF0000000000001ull, "-snan"},
};
static const size_t s_fp64_to_i_cases_n =
    sizeof(s_fp64_to_i_cases) / sizeof(s_fp64_to_i_cases[0]);

typedef struct int32_case {
    int32_t v;
    const char *name;
} int32_case;
static const int32_case s_int32_cases[] = {
    {0, "0"},
    {1, "+1"},
    {-1, "-1"},
    {2, "+2"},
    {-2, "-2"},
    {(1 << 22) - 1, "+2^22-1"},
    {(1 << 23), "+2^23"},
    {(1 << 23) + 1, "+2^23+1"},
    {(1 << 24), "+2^24 (exact fp32 boundary)"},
    {(1 << 24) + 1, "+2^24+1 (first inexact)"},
    {(1 << 24) + 2, "+2^24+2"},
    {(1 << 24) + 3, "+2^24+3 (halfway RNE)"},
    {-(1 << 24), "-2^24"},
    {-(1 << 24) - 1, "-2^24-1"},
    {-(1 << 24) - 3, "-2^24-3"},
    {0x7FFFFFFF, "INT32_MAX"},
    {(int32_t)0x80000000, "INT32_MIN"},
    {0x7FFFFFFE, "INT32_MAX-1"},
    {0x01234567, "+0x01234567 (mixed)"},
    {(int32_t)0xFEDCBA98, "-0x01234568 (mixed)"},
};
static const size_t s_int32_cases_n =
    sizeof(s_int32_cases) / sizeof(s_int32_cases[0]);

typedef struct int64_case {
    int64_t v;
    const char *name;
} int64_case;
static const int64_case s_int64_cases[] = {
    {0, "0"},
    {1, "+1"},
    {-1, "-1"},
    {(1LL << 23), "+2^23"},
    {(1LL << 24), "+2^24"},
    {(1LL << 24) + 1, "+2^24+1"},
    {(1LL << 52) - 1, "+2^52-1"},
    {(1LL << 52), "+2^52"},
    {(1LL << 53), "+2^53 (fp64 exact boundary)"},
    {(1LL << 53) + 1, "+2^53+1 (first fp64 inexact)"},
    {(1LL << 53) + 3, "+2^53+3 (halfway)"},
    {-((1LL << 52) - 1), "-(2^52-1)"},
    {-(1LL << 53) - 1, "-2^53-1"},
    {-(1LL << 53) - 3, "-2^53-3"},
    {(1LL << 62), "+2^62"},
    {0x7FFFFFFFFFFFFFFFLL, "INT64_MAX"},
    {0x7FFFFFFFFFFFFFFELL, "INT64_MAX-1"},
    {0x7FFFFFFFFFFFFC00LL, "INT64_MAX-1023"},
    {(int64_t)0x8000000000000000LL, "INT64_MIN"},
    {(int64_t)0x8000000000000001LL, "INT64_MIN+1"},
    {(int64_t)0x8000000000000400LL, "INT64_MIN+1024"},
    {0x0123456789ABCDEFLL, "+0x0123456789ABCDEF"},
    {(int64_t)0xFEDCBA9876543211LL, "-0x0123456789ABCDEF (neg)"},
    // Integer values near FP32 rounding boundaries; all are exactly representable in FP64.
    {(1LL << 24) + 3, "+2^24+3 fp32 halfway"},
    {(1LL << 40) + 0x1234567, "+2^40+mixed"},
    {-((1LL << 40) + 0x1234567), "-2^40-mixed"},
};
static const size_t s_int64_cases_n =
    sizeof(s_int64_cases) / sizeof(s_int64_cases[0]);

static uint64_t s_edge_passed = 0;
static uint64_t s_edge_failed = 0;

static const char *fp32_case_name(uint32_t bits) {
    for (size_t i = 0; i < s_fp32_cases_n; i++) {
        if (s_fp32_cases[i].bits == bits) {
            return s_fp32_cases[i].name;
        }
    }
    return "?";
}

static const char *fp16_case_name(uint16_t bits) {
    for (size_t i = 0; i < s_fp16_cases_n; i++) {
        if (s_fp16_cases[i].bits == bits) {
            return s_fp16_cases[i].name;
        }
    }
    return "?";
}

static const char *fp64_case_name(uint64_t bits) {
    for (size_t i = 0; i < s_fp64_cases_n; i++) {
        if (s_fp64_cases[i].bits == bits) {
            return s_fp64_cases[i].name;
        }
    }
    return "?";
}

static const char *fp32_to_i_case_name(uint32_t bits) {
    for (size_t i = 0; i < s_fp32_to_i_cases_n; i++) {
        if (s_fp32_to_i_cases[i].bits == bits) {
            return s_fp32_to_i_cases[i].name;
        }
    }
    return "?";
}

static const char *fp64_to_i_case_name(uint64_t bits) {
    for (size_t i = 0; i < s_fp64_to_i_cases_n; i++) {
        if (s_fp64_to_i_cases[i].bits == bits) {
            return s_fp64_to_i_cases[i].name;
        }
    }
    return "?";
}

static void edge_check_binop_fp32(const char *op_name,
                                  uint32_t (*hw)(uint32_t, uint32_t, uint32_t *),
                                  uint32_t (*sw)(uint32_t, uint32_t, uint32_t *),
                                  uint32_t a, uint32_t b, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw(a, b, &hw_mxcsr);
    uint32_t sw_r = sw(a, b, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s %s(%s) mxcsr=0x%04x -> hw=0x%08x/0x%04x sw=0x%08x/0x%04x\n",
                   op_name, fp32_case_name(a), fp32_case_name(b), init_mxcsr, hw_r,
                   hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_cmp_fp32(uint32_t a, uint32_t b, uint32_t imm, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw_fp32_cmp(a, b, imm, &hw_mxcsr);
    uint32_t sw_r = fpemu_fp32_cmp(a, b, imm, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("cmpss %s(%s, imm=0x%x) mxcsr=0x%04x -> "
                   "hw=0x%08x/0x%04x sw=0x%08x/0x%04x\n",
                   fp32_case_name(a), fp32_case_name(b), imm, init_mxcsr, hw_r,
                   hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_comi_fp32(const char *op_name,
                                 uint32_t (*hw)(uint32_t, uint32_t, uint32_t *),
                                 uint32_t (*sw)(uint32_t, uint32_t, uint32_t *),
                                 uint32_t a, uint32_t b, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw(a, b, &hw_mxcsr);
    uint32_t sw_r = sw(a, b, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s %s(%s) mxcsr=0x%04x -> "
                   "hw_flags=0x%03x/0x%04x sw_flags=0x%03x/0x%04x\n",
                   op_name, fp32_case_name(a), fp32_case_name(b), init_mxcsr,
                   hw_r, hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_triop_fp32(const char *op_name,
                                  uint32_t (*hw)(uint32_t, uint32_t, uint32_t, uint32_t *),
                                  uint32_t (*sw)(uint32_t, uint32_t, uint32_t, uint32_t *),
                                  uint32_t a, uint32_t b, uint32_t c, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw(a, b, c, &hw_mxcsr);
    uint32_t sw_r = sw(a, b, c, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s %s(%s,%s) mxcsr=0x%04x -> hw=0x%08x/0x%04x "
                   "sw=0x%08x/0x%04x\n",
                   op_name, fp32_case_name(a), fp32_case_name(b),
                   fp32_case_name(c), init_mxcsr, hw_r, hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_unary_fp32(const char *op_name, uint32_t (*hw)(uint32_t, uint32_t *),
                                  uint32_t (*sw)(uint32_t, uint32_t *),
                                  uint32_t a, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw(a, &hw_mxcsr);
    uint32_t sw_r = sw(a, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s(%s) mxcsr=0x%04x -> hw=0x%08x/0x%04x sw=0x%08x/0x%04x\n",
                   op_name, fp32_case_name(a), init_mxcsr, hw_r, hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_round_fp32(uint32_t a, uint32_t imm, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw_fp32_round(a, imm, &hw_mxcsr);
    uint32_t sw_r = fpemu_fp32_round(a, imm, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("roundss(%s, imm=0x%x) mxcsr=0x%04x -> "
                   "hw=0x%08x/0x%04x sw=0x%08x/0x%04x\n",
                   fp32_case_name(a), imm, init_mxcsr, hw_r, hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_binop_fp64(const char *op_name,
                                  uint64_t (*hw)(uint64_t, uint64_t, uint32_t *),
                                  uint64_t (*sw)(uint64_t, uint64_t, uint32_t *),
                                  uint64_t a, uint64_t b, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint64_t hw_r = hw(a, b, &hw_mxcsr);
    uint64_t sw_r = sw(a, b, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s %s(%s) mxcsr=0x%04x -> hw=0x%016llx/0x%04x "
                   "sw=0x%016llx/0x%04x\n",
                   op_name, fp64_case_name(a), fp64_case_name(b), init_mxcsr,
                   (unsigned long long)hw_r, hw_mxcsr, (unsigned long long)sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_cmp_fp64(uint64_t a, uint64_t b, uint32_t imm, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint64_t hw_r = hw_fp64_cmp(a, b, imm, &hw_mxcsr);
    uint64_t sw_r = fpemu_fp64_cmp(a, b, imm, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("cmpsd %s(%s, imm=0x%x) mxcsr=0x%04x -> "
                   "hw=0x%016llx/0x%04x sw=0x%016llx/0x%04x\n",
                   fp64_case_name(a), fp64_case_name(b), imm, init_mxcsr,
                   (unsigned long long)hw_r, hw_mxcsr, (unsigned long long)sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_comi_fp64(const char *op_name,
                                 uint32_t (*hw)(uint64_t, uint64_t, uint32_t *),
                                 uint32_t (*sw)(uint64_t, uint64_t, uint32_t *),
                                 uint64_t a, uint64_t b, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw(a, b, &hw_mxcsr);
    uint32_t sw_r = sw(a, b, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s %s(%s) mxcsr=0x%04x -> "
                   "hw_flags=0x%03x/0x%04x sw_flags=0x%03x/0x%04x\n",
                   op_name, fp64_case_name(a), fp64_case_name(b), init_mxcsr,
                   hw_r, hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_triop_fp64(const char *op_name,
                                  uint64_t (*hw)(uint64_t, uint64_t, uint64_t, uint32_t *),
                                  uint64_t (*sw)(uint64_t, uint64_t, uint64_t, uint32_t *),
                                  uint64_t a, uint64_t b, uint64_t c, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint64_t hw_r = hw(a, b, c, &hw_mxcsr);
    uint64_t sw_r = sw(a, b, c, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s %s(%s,%s) mxcsr=0x%04x -> hw=0x%016llx/0x%04x "
                   "sw=0x%016llx/0x%04x\n",
                   op_name, fp64_case_name(a), fp64_case_name(b),
                   fp64_case_name(c), init_mxcsr, (unsigned long long)hw_r, hw_mxcsr,
                   (unsigned long long)sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_unary_fp64(const char *op_name, uint64_t (*hw)(uint64_t, uint32_t *),
                                  uint64_t (*sw)(uint64_t, uint32_t *),
                                  uint64_t a, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint64_t hw_r = hw(a, &hw_mxcsr);
    uint64_t sw_r = sw(a, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s(%s) mxcsr=0x%04x -> hw=0x%016llx/0x%04x "
                   "sw=0x%016llx/0x%04x\n",
                   op_name, fp64_case_name(a), init_mxcsr, (unsigned long long)hw_r,
                   hw_mxcsr, (unsigned long long)sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_round_fp64(uint64_t a, uint32_t imm, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint64_t hw_r = hw_fp64_round(a, imm, &hw_mxcsr);
    uint64_t sw_r = fpemu_fp64_round(a, imm, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("roundsd(%s, imm=0x%x) mxcsr=0x%04x -> "
                   "hw=0x%016llx/0x%04x sw=0x%016llx/0x%04x\n",
                   fp64_case_name(a), imm, init_mxcsr, (unsigned long long)hw_r,
                   hw_mxcsr, (unsigned long long)sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_cvt_fp32_fp64(uint32_t a, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint64_t hw_r = hw_fp32_to_fp64(a, &hw_mxcsr);
    uint64_t sw_r = fpemu_fp32_to_fp64(a, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("cvtss2sd %s mxcsr=0x%04x -> hw=0x%016llx/0x%04x "
                   "sw=0x%016llx/0x%04x\n",
                   fp32_case_name(a), init_mxcsr, (unsigned long long)hw_r, hw_mxcsr,
                   (unsigned long long)sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_cvt_fp16_fp32(uint16_t a, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw_fp16_to_fp32(a, &hw_mxcsr);
    uint32_t sw_r = fpemu_fp16_to_fp32(a, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("vcvtph2ps_lane %s(0x%04x) mxcsr=0x%04x -> "
                   "hw=0x%08x/0x%04x sw=0x%08x/0x%04x\n",
                   fp16_case_name(a), a, init_mxcsr, hw_r, hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_cvt_fp32_fp16(uint32_t a, uint32_t imm, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint16_t hw_r = hw_fp32_to_fp16(a, imm, &hw_mxcsr);
    uint16_t sw_r = fpemu_fp32_to_fp16(a, imm, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("vcvtps2ph_lane %s imm=0x%x mxcsr=0x%04x -> "
                   "hw=0x%04x/0x%04x sw=0x%04x/0x%04x\n",
                   fp32_case_name(a), imm, init_mxcsr, hw_r, hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_int32_fp32(int32_t a, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw_int32_to_fp32(a, &hw_mxcsr);
    uint32_t sw_r = fpemu_int32_to_fp32(a, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("cvtsi2ss_i32 %d mxcsr=0x%04x -> hw=0x%08x/0x%04x "
                   "sw=0x%08x/0x%04x\n",
                   a, init_mxcsr, hw_r, hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_int64_fp32(int64_t a, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw_int64_to_fp32(a, &hw_mxcsr);
    uint32_t sw_r = fpemu_int64_to_fp32(a, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("cvtsi2ss_i64 %lld mxcsr=0x%04x -> hw=0x%08x/0x%04x "
                   "sw=0x%08x/0x%04x\n",
                   (long long)a, init_mxcsr, hw_r, hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_int32_fp64(int32_t a, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint64_t hw_r = hw_int32_to_fp64(a, &hw_mxcsr);
    uint64_t sw_r = fpemu_int32_to_fp64(a, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("cvtsi2sd_i32 %d mxcsr=0x%04x -> hw=0x%016llx/0x%04x "
                   "sw=0x%016llx/0x%04x\n",
                   a, init_mxcsr, (unsigned long long)hw_r, hw_mxcsr,
                   (unsigned long long)sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_int64_fp64(int64_t a, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint64_t hw_r = hw_int64_to_fp64(a, &hw_mxcsr);
    uint64_t sw_r = fpemu_int64_to_fp64(a, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("cvtsi2sd_i64 %lld mxcsr=0x%04x -> hw=0x%016llx/0x%04x "
                   "sw=0x%016llx/0x%04x\n",
                   (long long)a, init_mxcsr, (unsigned long long)hw_r, hw_mxcsr,
                   (unsigned long long)sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_cvt_fp64_fp32(uint64_t a, uint32_t init_mxcsr) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = hw_fp64_to_fp32(a, &hw_mxcsr);
    uint32_t sw_r = fpemu_fp64_to_fp32(a, &sw_mxcsr);
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("cvtsd2ss %s (0x%016llx) mxcsr=0x%04x -> hw=0x%08x/0x%04x "
                   "sw=0x%08x/0x%04x\n",
                   fp64_case_name(a), (unsigned long long)a, init_mxcsr, hw_r,
                   hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_fp32_to_i32(uint32_t a, uint32_t init_mxcsr, bool trunc) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = trunc ? hw_fp32_to_i32_trunc(a, &hw_mxcsr)
                          : hw_fp32_to_i32(a, &hw_mxcsr);
    uint32_t sw_r = trunc ? fpemu_fp32_to_i32_trunc(a, &sw_mxcsr)
                          : fpemu_fp32_to_i32(a, &sw_mxcsr);
    const char *op = trunc ? "cvtss2si32_trunc" : "cvtss2si32";
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s %s(0x%08x) mxcsr=0x%04x -> hw=0x%08x/0x%04x "
                   "sw=0x%08x/0x%04x\n",
                   op, fp32_to_i_case_name(a), a, init_mxcsr, hw_r, hw_mxcsr,
                   sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_fp32_to_i64(uint32_t a, uint32_t init_mxcsr, bool trunc) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint64_t hw_r = trunc ? hw_fp32_to_i64_trunc(a, &hw_mxcsr)
                          : hw_fp32_to_i64(a, &hw_mxcsr);
    uint64_t sw_r = trunc ? fpemu_fp32_to_i64_trunc(a, &sw_mxcsr)
                          : fpemu_fp32_to_i64(a, &sw_mxcsr);
    const char *op = trunc ? "cvtss2si64_trunc" : "cvtss2si64";
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s %s(0x%08x) mxcsr=0x%04x -> hw=0x%016llx/0x%04x "
                   "sw=0x%016llx/0x%04x\n",
                   op, fp32_to_i_case_name(a), a, init_mxcsr,
                   (unsigned long long)hw_r, hw_mxcsr, (unsigned long long)sw_r,
                   sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_fp64_to_i32(uint64_t a, uint32_t init_mxcsr, bool trunc) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint32_t hw_r = trunc ? hw_fp64_to_i32_trunc(a, &hw_mxcsr)
                          : hw_fp64_to_i32(a, &hw_mxcsr);
    uint32_t sw_r = trunc ? fpemu_fp64_to_i32_trunc(a, &sw_mxcsr)
                          : fpemu_fp64_to_i32(a, &sw_mxcsr);
    const char *op = trunc ? "cvtsd2si32_trunc" : "cvtsd2si32";
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s %s(0x%016llx) mxcsr=0x%04x -> hw=0x%08x/0x%04x "
                   "sw=0x%08x/0x%04x\n",
                   op, fp64_to_i_case_name(a), (unsigned long long)a, init_mxcsr,
                   hw_r, hw_mxcsr, sw_r, sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static void edge_check_fp64_to_i64(uint64_t a, uint32_t init_mxcsr, bool trunc) {
    uint32_t hw_mxcsr = init_mxcsr;
    uint32_t sw_mxcsr = init_mxcsr;
    uint64_t hw_r = trunc ? hw_fp64_to_i64_trunc(a, &hw_mxcsr)
                          : hw_fp64_to_i64(a, &hw_mxcsr);
    uint64_t sw_r = trunc ? fpemu_fp64_to_i64_trunc(a, &sw_mxcsr)
                          : fpemu_fp64_to_i64(a, &sw_mxcsr);
    const char *op = trunc ? "cvtsd2si64_trunc" : "cvtsd2si64";
    if ((hw_r != sw_r) || (hw_mxcsr != sw_mxcsr)) {
        if (s_edge_failed < MAX_PRINT) {
            printf("%s %s(0x%016llx) mxcsr=0x%04x -> hw=0x%016llx/0x%04x "
                   "sw=0x%016llx/0x%04x\n",
                   op, fp64_to_i_case_name(a), (unsigned long long)a, init_mxcsr,
                   (unsigned long long)hw_r, hw_mxcsr, (unsigned long long)sw_r,
                   sw_mxcsr);
        }
        s_edge_failed++;
    } else {
        s_edge_passed++;
    }
}

static bool run_edge_tests(void) {
    // For each control-flag combination (rounding mode x FTZ x DAZ), run with both a clean MXCSR
    // (no sticky flags) and with every exception flag pre-set. Clean MXCSR surfaces software
    // paths that fail to raise flags; sticky pre-set confirms the software never clears flags it
    // shouldn't touch.
    const uint32_t stickies[] = {
        0,
        FPEMU_MXCSR_IE | FPEMU_MXCSR_DE | FPEMU_MXCSR_ZE | FPEMU_MXCSR_OE |
            FPEMU_MXCSR_UE | FPEMU_MXCSR_PE,
    };
    for (size_t s = 0; s < sizeof(stickies) / sizeof(stickies[0]); s++) {
        for (uint32_t rm = 0; rm < 4; rm++) {
            for (uint32_t ftz = 0; ftz <= 1; ftz++) {
                for (uint32_t daz = 0; daz <= 1; daz++) {
                    uint32_t mxcsr = FPEMU_MXCSR_DEFAULT | (rm << 13) |
                                     (ftz ? FPEMU_MXCSR_FTZ : 0) |
                                     (daz ? FPEMU_MXCSR_DAZ : 0) | stickies[s];
                    for (size_t i = 0; i < s_fp32_cases_n; i++) {
                        for (size_t j = 0; j < s_fp32_cases_n; j++) {
                            uint32_t a = s_fp32_cases[i].bits;
                            uint32_t b = s_fp32_cases[j].bits;
                            edge_check_binop_fp32("add", hw_fp32_add, fpemu_fp32_add, a, b, mxcsr);
                            edge_check_binop_fp32("sub", hw_fp32_sub, fpemu_fp32_sub, a, b, mxcsr);
                            edge_check_binop_fp32("mul", hw_fp32_mul, fpemu_fp32_mul, a, b, mxcsr);
                            edge_check_binop_fp32("div", hw_fp32_div, fpemu_fp32_div, a, b, mxcsr);
                            edge_check_binop_fp32("min", hw_fp32_min, fpemu_fp32_min, a, b, mxcsr);
                            edge_check_binop_fp32("max", hw_fp32_max, fpemu_fp32_max, a, b, mxcsr);
                            edge_check_comi_fp32("comiss", hw_fp32_comi, fpemu_fp32_comi, a, b,
                                                 mxcsr);
                            edge_check_comi_fp32("ucomiss", hw_fp32_ucomi, fpemu_fp32_ucomi, a, b,
                                                 mxcsr);
                            for (uint32_t imm = 0; imm < 32; imm++) {
                                edge_check_cmp_fp32(a, b, imm, mxcsr);
                            }
                        }
                    }
                    for (size_t i = 0; i < s_fp32_cases_n; i++) {
                        for (size_t j = 0; j < s_fp32_cases_n; j++) {
                            for (size_t k = 0; k < s_fp32_cases_n; k++) {
                                uint32_t a = s_fp32_cases[i].bits;
                                uint32_t b = s_fp32_cases[j].bits;
                                uint32_t c = s_fp32_cases[k].bits;
                                edge_check_triop_fp32("fmadd", hw_fp32_fmadd, fpemu_fp32_fmadd,
                                                      a, b, c, mxcsr);
                                edge_check_triop_fp32("fmsub", hw_fp32_fmsub, fpemu_fp32_fmsub,
                                                      a, b, c, mxcsr);
                                edge_check_triop_fp32("fnmadd", hw_fp32_fnmadd, fpemu_fp32_fnmadd,
                                                      a, b, c, mxcsr);
                                edge_check_triop_fp32("fnmsub", hw_fp32_fnmsub, fpemu_fp32_fnmsub,
                                                      a, b, c, mxcsr);
                            }
                        }
                    }
                    for (size_t i = 0; i < s_fp64_cases_n; i++) {
                        for (size_t j = 0; j < s_fp64_cases_n; j++) {
                            uint64_t a = s_fp64_cases[i].bits;
                            uint64_t b = s_fp64_cases[j].bits;
                            edge_check_binop_fp64("add", hw_fp64_add, fpemu_fp64_add, a, b, mxcsr);
                            edge_check_binop_fp64("sub", hw_fp64_sub, fpemu_fp64_sub, a, b, mxcsr);
                            edge_check_binop_fp64("mul", hw_fp64_mul, fpemu_fp64_mul, a, b, mxcsr);
                            edge_check_binop_fp64("div", hw_fp64_div, fpemu_fp64_div, a, b, mxcsr);
                            edge_check_binop_fp64("min", hw_fp64_min, fpemu_fp64_min, a, b, mxcsr);
                            edge_check_binop_fp64("max", hw_fp64_max, fpemu_fp64_max, a, b, mxcsr);
                            edge_check_comi_fp64("comisd", hw_fp64_comi, fpemu_fp64_comi, a, b,
                                                 mxcsr);
                            edge_check_comi_fp64("ucomisd", hw_fp64_ucomi, fpemu_fp64_ucomi, a, b,
                                                 mxcsr);
                            for (uint32_t imm = 0; imm < 32; imm++) {
                                edge_check_cmp_fp64(a, b, imm, mxcsr);
                            }
                        }
                    }
                    for (size_t i = 0; i < s_fp64_cases_n; i++) {
                        for (size_t j = 0; j < s_fp64_cases_n; j++) {
                            for (size_t k = 0; k < s_fp64_cases_n; k++) {
                                uint64_t a = s_fp64_cases[i].bits;
                                uint64_t b = s_fp64_cases[j].bits;
                                uint64_t c = s_fp64_cases[k].bits;
                                edge_check_triop_fp64("fmadd", hw_fp64_fmadd, fpemu_fp64_fmadd,
                                                      a, b, c, mxcsr);
                                edge_check_triop_fp64("fmsub", hw_fp64_fmsub, fpemu_fp64_fmsub,
                                                      a, b, c, mxcsr);
                                edge_check_triop_fp64("fnmadd", hw_fp64_fnmadd, fpemu_fp64_fnmadd,
                                                      a, b, c, mxcsr);
                                edge_check_triop_fp64("fnmsub", hw_fp64_fnmsub, fpemu_fp64_fnmsub,
                                                      a, b, c, mxcsr);
                            }
                        }
                    }
                    for (size_t i = 0; i < s_fp32_cases_n; i++) {
                        edge_check_unary_fp32("sqrt", hw_fp32_sqrt, fpemu_fp32_sqrt,
                                              s_fp32_cases[i].bits, mxcsr);
                        for (uint32_t imm = 0; imm < 16; imm++) {
                            edge_check_round_fp32(s_fp32_cases[i].bits, imm, mxcsr);
                        }
                        edge_check_cvt_fp32_fp64(s_fp32_cases[i].bits, mxcsr);
                        for (uint32_t imm = 0; imm < 8; imm++) {
                            edge_check_cvt_fp32_fp16(s_fp32_cases[i].bits, imm, mxcsr);
                        }
                    }
                    for (size_t i = 0; i < s_fp64_cases_n; i++) {
                        edge_check_unary_fp64("sqrt", hw_fp64_sqrt, fpemu_fp64_sqrt,
                                              s_fp64_cases[i].bits, mxcsr);
                        for (uint32_t imm = 0; imm < 16; imm++) {
                            edge_check_round_fp64(s_fp64_cases[i].bits, imm, mxcsr);
                        }
                    }
                    for (size_t i = 0; i < s_fp64_cases_n; i++) {
                        edge_check_cvt_fp64_fp32(s_fp64_cases[i].bits, mxcsr);
                    }
                    for (size_t i = 0; i < s_fp16_cases_n; i++) {
                        edge_check_cvt_fp16_fp32(s_fp16_cases[i].bits, mxcsr);
                    }
                    for (size_t i = 0; i < s_fp32_to_i_cases_n; i++) {
                        uint32_t a = s_fp32_to_i_cases[i].bits;
                        edge_check_fp32_to_i32(a, mxcsr, false);
                        edge_check_fp32_to_i32(a, mxcsr, true);
                        edge_check_fp32_to_i64(a, mxcsr, false);
                        edge_check_fp32_to_i64(a, mxcsr, true);
                    }
                    for (size_t i = 0; i < s_fp64_to_i_cases_n; i++) {
                        uint64_t a = s_fp64_to_i_cases[i].bits;
                        edge_check_fp64_to_i32(a, mxcsr, false);
                        edge_check_fp64_to_i32(a, mxcsr, true);
                        edge_check_fp64_to_i64(a, mxcsr, false);
                        edge_check_fp64_to_i64(a, mxcsr, true);
                    }
                    for (size_t i = 0; i < s_int32_cases_n; i++) {
                        edge_check_int32_fp32(s_int32_cases[i].v, mxcsr);
                        edge_check_int32_fp64(s_int32_cases[i].v, mxcsr);
                    }
                    for (size_t i = 0; i < s_int64_cases_n; i++) {
                        edge_check_int64_fp32(s_int64_cases[i].v, mxcsr);
                        edge_check_int64_fp64(s_int64_cases[i].v, mxcsr);
                    }
                }
            }
        }
    }
    if (!s_quiet_success || s_edge_failed != 0) {
        printf("edge: %llu passed, %llu failed\n",
               (unsigned long long)s_edge_passed, (unsigned long long)s_edge_failed);
    }
    return s_edge_failed == 0;
}

static uint64_t pow10_u64(unsigned long e) {
    uint64_t loops = 1;
    while (e--) {
        loops *= 10;
    }
    return loops;
}

static bool parse_u64(const char *s, uint64_t *p_value) {
    if (!*s) {
        return false;
    }
    uint64_t value = 0;
    for (; *s; s++) {
        if ((*s < '0') || (*s > '9')) {
            return false;
        }
        uint64_t digit = (uint64_t)(*s - '0');
        if (value > (UINT64_MAX - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    *p_value = value;
    return true;
}

static bool parse_cli_u64(const char *name, const char *s, uint64_t *p_value) {
    if (parse_u64(s, p_value)) {
        return true;
    }
    fprintf(stderr, "ERROR: invalid %s '%s'\n", name, s);
    return false;
}

static bool parse_cli_u32_range(const char *name, const char *s,
                                uint32_t max, uint32_t *p_value) {
    uint64_t value;
    if (!parse_cli_u64(name, s, &value)) {
        return false;
    }
    if (value > max) {
        fprintf(stderr, "ERROR: invalid %s '%s' (must be 0..%u)\n", name, s, max);
        return false;
    }
    *p_value = (uint32_t)value;
    return true;
}

static int finish_status(const char *log_path, bool ok, const char *fmt, ...) {
    if (log_path != NULL) {
        FILE *f = fopen(log_path, "w");
        if (f == NULL) {
            fprintf(stderr, "ERROR: could not open status log '%s'\n", log_path);
            return 2;
        }
        fprintf(f, "%s ", ok ? "DONE" : "FAILED");
        va_list ap;
        va_start(ap, fmt);
        vfprintf(f, fmt, ap);
        va_end(ap);
        fputc('\n', f);
        if (fclose(f) != 0) {
            fprintf(stderr, "ERROR: could not write status log '%s'\n", log_path);
            return 2;
        }
    }
    return ok ? 0 : 1;
}

static int usage(const char *argv0) {
    fprintf(stderr,
        "usage:\n"
        "  %s rand <log10-cases> <seed> [log]\n"
        "  %s edge [log]\n"
        "  %s exhaustive <op> <rm> <daz> <ftz> <imm> <index> <count> [log]\n"
        "                               # exhaustive single-input op over one "
        "control setting\n",
        argv0, argv0, argv0);
    return 2;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        return usage(argv[0]);
    }
    if (!strcmp(argv[1], "edge")) {
        if ((argc != 2) && (argc != 3)) {
            return usage(argv[0]);
        }
        const char *log_path = (argc == 3) ? argv[2] : NULL;
        s_quiet_success = log_path != NULL;
        bool ok = run_edge_tests();
        return finish_status(log_path, ok, "edge passed=%llu failed=%llu",
                             (unsigned long long)s_edge_passed,
                             (unsigned long long)s_edge_failed);
    }
    if (!strcmp(argv[1], "exhaustive")) {
        if ((argc != 9) && (argc != 10)) {
            return usage(argv[0]);
        }
        test_op op;
        if (!parse_exhaustive_op(argv[2], &op)) {
            fprintf(stderr, "ERROR: unsupported exhaustive op '%s'\n", argv[2]);
            return 2;
        }
        uint32_t rm, daz, ftz, imm;
        uint32_t shard_index, shard_count;
        if (!parse_cli_u32_range("rm", argv[3], 3, &rm) ||
            !parse_cli_u32_range("daz", argv[4], 1, &daz) ||
            !parse_cli_u32_range("ftz", argv[5], 1, &ftz) ||
            !parse_cli_u32_range("imm", argv[6], 31, &imm) ||
            !parse_cli_u32_range("shard index", argv[7], 9999, &shard_index) ||
            !parse_cli_u32_range("shard count", argv[8], 10000, &shard_count)) {
            return 2;
        }
        const char *log_path = (argc == 10) ? argv[9] : NULL;
        if ((shard_count == 0) || (shard_index >= shard_count)) {
            fprintf(stderr, "ERROR: invalid shard index/count '%s/%s' "
                    "(index must be less than count, and count must be nonzero)\n",
                    argv[7], argv[8]);
            return 2;
        }
        s_quiet_success = log_path != NULL;
        if (!s_quiet_success) {
            printf("Running exhaustive %s rm=%u daz=%u ftz=%u imm=%u shard %llu/%llu...\n",
                   opcode_name(op), rm, daz, ftz, imm,
                   (unsigned long long)shard_index, (unsigned long long)shard_count);
        }
        bool ok = run_exhaustive(op, rm, daz != 0, ftz != 0, imm, shard_index,
                                 shard_count);
        return finish_status(log_path, ok,
                             "exhaustive op=%s rm=%u daz=%u ftz=%u imm=%u "
                             "shard=%llu/%llu",
                             opcode_name(op), rm, daz, ftz, imm,
                             (unsigned long long)shard_index,
                             (unsigned long long)shard_count);
    }
    if (!strcmp(argv[1], "rand")) {
        if ((argc != 4) && (argc != 5)) {
            return usage(argv[0]);
        }
        uint32_t log10_cases;
        uint64_t seed;
        if (!parse_cli_u32_range("log10-cases", argv[2], 20, &log10_cases) ||
            !parse_cli_u64("seed", argv[3], &seed)) {
            return 2;
        }
        const char *log_path = (argc == 5) ? argv[4] : NULL;
        uint64_t loops = pow10_u64(log10_cases);
        s_quiet_success = log_path != NULL;
        if (!s_quiet_success) {
            printf("Running %llu cases (seed=%llu)...\n",
                   (unsigned long long)loops, (unsigned long long)seed);
        }
        bool ok = run_random_tests(loops, seed);
        return finish_status(log_path, ok,
                             "rand log10_cases=%u seed=%llu cases=%llu",
                             log10_cases, (unsigned long long)seed,
                             (unsigned long long)loops);
    }
    return usage(argv[0]);
}
