// fpemu (https://github.com/mjcraighead/fpemu)
// Copyright (c) 2025-2026 Matt Craighead
// SPDX-License-Identifier: MIT

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "fpemu.h"

#define NORETURN __attribute__((noreturn))
#define NO_PROFILE __attribute__((no_profile_instrument_function))

#define FPEMU_ASSERT(cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            fpemu_assert_failed(__func__, fmt, ##__VA_ARGS__); \
        } \
    } while (0)

#define FP32_SIGN_MASK    0x80000000u
#define FP32_ABS_MASK     0x7FFFFFFFu
#define FP32_EXP_MASK     0x7F800000u
#define FP32_FRAC_MASK    0x007FFFFFu
#define FP32_IMPLICIT_BIT 0x00800000u
#define FP32_MAX_FINITE   0x7F7FFFFFu
#define FP32_INF          0x7F800000u
#define FP32_QUIET_MASK   0x00400000u
#define FP32_QNAN         0x7FC00000u
#define FP32_DEFAULT_NAN  0xFFC00000u

#define FP64_SIGN_MASK    0x8000000000000000ull
#define FP64_ABS_MASK     0x7FFFFFFFFFFFFFFFull
#define FP64_EXP_MASK     0x7FF0000000000000ull
#define FP64_FRAC_MASK    0x000FFFFFFFFFFFFFull
#define FP64_IMPLICIT_BIT 0x0010000000000000ull
#define FP64_MAX_FINITE   0x7FEFFFFFFFFFFFFFull
#define FP64_INF          0x7FF0000000000000ull
#define FP64_QUIET_MASK   0x0008000000000000ull
#define FP64_QNAN         0x7FF8000000000000ull
#define FP64_DEFAULT_NAN  0xFFF8000000000000ull

#define FP32_IS_INF(x) (((x) & FP32_ABS_MASK) == FP32_INF)
#define FP32_IS_NAN(x) (((x) & FP32_ABS_MASK) > FP32_INF)
#define FP32_IS_SNAN(x) (FP32_IS_NAN(x) && !((x) & FP32_QUIET_MASK))
#define FP32_IS_SUBNORMAL(x) (!((x) & FP32_EXP_MASK) && ((x) & FP32_FRAC_MASK))

#define FP64_IS_INF(x) (((x) & FP64_ABS_MASK) == FP64_INF)
#define FP64_IS_NAN(x) (((x) & FP64_ABS_MASK) > FP64_INF)
#define FP64_IS_SNAN(x) (FP64_IS_NAN(x) && !((x) & FP64_QUIET_MASK))
#define FP64_IS_SUBNORMAL(x) (!((x) & FP64_EXP_MASK) && ((x) & FP64_FRAC_MASK))

static NORETURN NO_PROFILE void fpemu_assert_failed(const char *function, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "fpemu assertion failed: %s: ", function);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

static void fpemu_check_mxcsr(const uint32_t *p_mxcsr) {
    FPEMU_ASSERT(p_mxcsr != NULL, "p_mxcsr is NULL");
    FPEMU_ASSERT((*p_mxcsr & ~FPEMU_MXCSR_DEFINED_MASK) == 0,
        "invalid MXCSR high bits: mxcsr=0x%08x", *p_mxcsr);
    FPEMU_ASSERT((*p_mxcsr & FPEMU_MXCSR_EXCEPTION_MASKS) == FPEMU_MXCSR_EXCEPTION_MASKS,
        "unmasked MXCSR exceptions are out of scope: mxcsr=0x%08x", *p_mxcsr);
}

static int clz_u128(__uint128_t v) {
    if (v >> 64) {
        return __builtin_clzll((uint64_t)(v >> 64));
    }
    return 64 + __builtin_clzll((uint64_t)v);
}

static __uint128_t align_right_with_sticky_u128(__uint128_t v, int shift) {
    FPEMU_ASSERT(shift > 0, "align_right_with_sticky_u128 shift=%d", shift);
    if (shift >= 128) {
        return (v != 0) ? 1ull : 0ull;
    }
    __uint128_t mask = ((__uint128_t)1 << shift) - 1;
    __uint128_t dropped = v & mask;
    v >>= shift;
    if (dropped != 0) {
        v |= 1ull;
    }
    return v;
}

static uint32_t round_increment(uint32_t rm, bool sign_negative, uint32_t mant, uint32_t guard,
                                uint32_t sticky) {
    if (rm == 0) {
        return (guard && (sticky || (mant & 1))) ? 1u : 0u;
    } else if (rm == 1) {
        return sign_negative ? (guard || sticky) : 0;
    } else if (rm == 2) {
        return sign_negative ? 0 : (guard || sticky);
    } else {
        return 0; // rm==3 (toward zero): never round up
    }
}

static bool quiet_binary_nan32(uint32_t a, uint32_t b, uint32_t *p_mxcsr, uint32_t *p_result) {
    if (FP32_IS_NAN(a)) {
        if (!(a & FP32_QUIET_MASK) || FP32_IS_SNAN(b)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        *p_result = a | FP32_QUIET_MASK;
        return true;
    }
    if (FP32_IS_NAN(b)) {
        if (!(b & FP32_QUIET_MASK)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        *p_result = b | FP32_QUIET_MASK;
        return true;
    }
    return false;
}

static bool quiet_binary_nan64(uint64_t a, uint64_t b, uint32_t *p_mxcsr, uint64_t *p_result) {
    if (FP64_IS_NAN(a)) {
        if (!(a & FP64_QUIET_MASK) || FP64_IS_SNAN(b)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        *p_result = a | FP64_QUIET_MASK;
        return true;
    }
    if (FP64_IS_NAN(b)) {
        if (!(b & FP64_QUIET_MASK)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        *p_result = b | FP64_QUIET_MASK;
        return true;
    }
    return false;
}

static bool fp32_ordered_less(uint32_t a, uint32_t b) {
    uint32_t abs_a = a & FP32_ABS_MASK;
    uint32_t abs_b = b & FP32_ABS_MASK;
    if (!abs_a && !abs_b) {
        return false;
    }
    bool sign_a = (a & FP32_SIGN_MASK) != 0;
    bool sign_b = (b & FP32_SIGN_MASK) != 0;
    if (sign_a != sign_b) {
        return sign_a;
    }
    if (sign_a) {
        return abs_a > abs_b;
    }
    return abs_a < abs_b;
}

static bool fp64_ordered_less(uint64_t a, uint64_t b) {
    uint64_t abs_a = a & FP64_ABS_MASK;
    uint64_t abs_b = b & FP64_ABS_MASK;
    if (!abs_a && !abs_b) {
        return false;
    }
    bool sign_a = (a & FP64_SIGN_MASK) != 0;
    bool sign_b = (b & FP64_SIGN_MASK) != 0;
    if (sign_a != sign_b) {
        return sign_a;
    }
    if (sign_a) {
        return abs_a > abs_b;
    }
    return abs_a < abs_b;
}

static bool fp32_ordered_equal(uint32_t a, uint32_t b) {
    if (!((a | b) & FP32_ABS_MASK)) {
        return true;
    }
    return a == b;
}

static bool fp64_ordered_equal(uint64_t a, uint64_t b) {
    if (!((a | b) & FP64_ABS_MASK)) {
        return true;
    }
    return a == b;
}

static uint32_t round_control_from_imm(uint32_t imm, uint32_t mxcsr) {
    return (imm & 4) ? ((mxcsr >> 13) & 3) : (imm & 3);
}

static bool cmp_predicate_signals(uint32_t pred) {
    switch (pred & 31) {
        case 1:
        case 2:
        case 5:
        case 6:
        case 9:
        case 10:
        case 13:
        case 14:
        case 16:
        case 19:
        case 20:
        case 23:
        case 24:
        case 27:
        case 28:
        case 31:
            return true;
        default:
            return false;
    }
}

static bool cmp_predicate_value(uint32_t pred, bool unordered, bool less, bool equal) {
    bool ordered = !unordered;
    bool greater = ordered && !less && !equal;
    switch (pred & 15) {
        case 0: return ordered && equal;
        case 1: return ordered && less;
        case 2: return ordered && (less || equal);
        case 3: return unordered;
        case 4: return unordered || !equal;
        case 5: return unordered || !less;
        case 6: return unordered || !(less || equal);
        case 7: return ordered;
        case 8: return unordered || equal;
        case 9: return unordered || less;
        case 10: return unordered || !greater;
        case 11: return false;
        case 12: return ordered && !equal;
        case 13: return ordered && !less;
        case 14: return greater;
        default: return true;
    }
}

static uint32_t comi_flags(bool unordered, bool less, bool equal) {
    if (unordered) {
        return FPEMU_EFLAGS_ZF | FPEMU_EFLAGS_PF | FPEMU_EFLAGS_CF;
    } else if (equal) {
        return FPEMU_EFLAGS_ZF;
    } else if (less) {
        return FPEMU_EFLAGS_CF;
    } else {
        return 0;
    }
}

static uint64_t round_to_fp64(__uint128_t mant_result, int32_t exp_result,
                              uint64_t sign_result, uint32_t *p_mxcsr) {
    uint32_t rm = (*p_mxcsr >> 13) & 3;
    if (!mant_result) {
        return (rm == 1) ? (sign_result | FP64_SIGN_MASK) : sign_result;
    }

    int clz = clz_u128(mant_result);
    // Mantissas entering round_to_fp64 should be <= bit 106 (top at 105 or below).
    // Keep this invariant locally to avoid undefined shifts.
    FPEMU_ASSERT(clz >= 22 && clz <= 127,
                 "round_to_fp64 clz=%d mant_result_hi=0x%016llx mant_result_lo=0x%016llx",
                 clz, (unsigned long long)(mant_result >> 64), (unsigned long long)mant_result);
    int shift0 = clz - 22;
    mant_result <<= shift0;
    exp_result += 1 - shift0;

    const int nshift = 53;
    uint64_t mant_n = (mant_result >> nshift) & 0x1FFFFFFFFFFFFFull;
    uint64_t guard_n = (mant_result >> (nshift - 1)) & 1ull;
    bool sticky_n = (mant_result & (((__uint128_t)1 << (nshift - 1)) - 1ull)) != 0ull;
    bool inexact_n = (guard_n | sticky_n) != 0;
    int32_t exp_n = exp_result;
    uint32_t inc_n = round_increment(rm, sign_result, (uint32_t)mant_n, (uint32_t)guard_n,
                                     (uint32_t)sticky_n);
    if (inc_n) {
        mant_n++;
        if (mant_n == 0x20000000000000ull) {
            mant_n = 0;
            exp_n++;
        }
    }
    bool tiny = (exp_n <= 0);

    if (exp_n >= 2047) {
        *p_mxcsr |= FPEMU_MXCSR_OE | FPEMU_MXCSR_PE;
        if (rm == 0) {
            return sign_result | FP64_INF;
        } else if (rm == 1) {
            return sign_result ? (sign_result | FP64_INF)
                               : (sign_result | FP64_MAX_FINITE);
        } else if (rm == 2) {
            return sign_result ? (sign_result | FP64_MAX_FINITE)
                               : (sign_result | FP64_INF);
        } else {
            return sign_result | FP64_MAX_FINITE;
        }
    }

    if (!tiny) {
        if (inexact_n) {
            *p_mxcsr |= FPEMU_MXCSR_PE;
        }
        return sign_result | ((uint64_t)exp_n << 52) | ((uint64_t)mant_n & FP64_FRAC_MASK);
    }

    if (*p_mxcsr & FPEMU_MXCSR_FTZ) {
        *p_mxcsr |= FPEMU_MXCSR_UE | FPEMU_MXCSR_PE;
        return sign_result;
    }

    int sshift = nshift + (1 - exp_result);
    uint64_t mant_s, guard_s;
    bool sticky_s;
    if (sshift > 127) {
        mant_s = 0;
        guard_s = 0;
        sticky_s = mant_result != 0;
    } else {
        mant_s = (mant_result >> sshift) & 0x1FFFFFFFFFFFFFull;
        guard_s = (mant_result >> (sshift - 1)) & 1ull;
        sticky_s = (mant_result & (((__uint128_t)1 << (sshift - 1)) - 1)) != 0;
    }
    bool inexact_s = (guard_s | sticky_s) != 0;
    uint32_t inc_s = round_increment(rm, sign_result, (uint32_t)mant_s, (uint32_t)guard_s,
                                     (uint32_t)sticky_s);
    if (inc_s) {
        // If subnormal rounding could carry into min-normal here, the earlier normal-precision
        // rounding would already have set exp_n=1 and made tiny false. Reaching this path
        // means our local invariants are wrong.
        FPEMU_ASSERT(mant_s != 0x1FFFFFFFFFFFFFull,
                     "round_to_fp64 unexpected subnormal carry rm=%u sign=0x%016llx exp_result=%d",
                     rm, (unsigned long long)sign_result, exp_result);
        mant_s++;
    }
    if (inexact_s) {
        *p_mxcsr |= FPEMU_MXCSR_UE | FPEMU_MXCSR_PE;
    }
    return sign_result | mant_s;
}

static uint64_t abs_i32_to_u64(int32_t a) {
    return (a < 0) ? (uint64_t)-(int64_t)a : (uint64_t)(uint32_t)a;
}

static uint64_t abs_i64_to_u64(int64_t a) {
    return (a < 0) ? (0ull - (uint64_t)a) : (uint64_t)a;
}

// Round a positive FP magnitude represented as mantissa * 2^exp into an unsigned integer, and
// report inexactness. `trunc` selects the truncating family (CVTT); otherwise current MXCSR
// rounding mode applies to nonzero discarded bits.
static uint64_t round_fp_to_uint(uint64_t mantissa, int32_t exp, bool sign_negative, bool trunc,
                                 uint32_t *p_mxcsr, bool *p_inexact, bool *p_overflow) {
    uint32_t rm = (*p_mxcsr >> 13) & 3;
    uint64_t rounded = 0;
    if (p_overflow) {
        *p_overflow = false;
    }
    if (exp >= 0) {
        if (exp >= 64) {
            rounded = 0;
            if (p_overflow) {
                *p_overflow = true;
            }
        } else {
            // For large positive shifts, left-shifting into the high bits can overflow
            // 64-bit integer math before we have a chance to range-check against the
            // destination integer width.
            if (mantissa > (UINT64_MAX >> exp)) {
                rounded = 0;
                if (p_overflow) {
                    *p_overflow = true;
                }
            } else {
                rounded = mantissa << exp;
            }
        }
        if (p_inexact) {
            *p_inexact = false;
        }
        return rounded;
    }

    int32_t shift = -exp;
    uint32_t guard = 0;
    uint32_t sticky = 0;
    if (shift >= 64) {
        // All nonzero mantissa bits are discarded, so the value is inexact if mantissa was
        // nonzero and never rounds up.
        rounded = 0;
        sticky = (mantissa != 0);
    } else {
        rounded = mantissa >> shift;
        if (shift > 0) {
            guard = (mantissa >> (shift - 1)) & 1;
        }
        if (shift > 1) {
            sticky = (mantissa & ((1ull << (shift - 1)) - 1)) != 0;
        }
    }

    bool inexact = (guard | sticky) != 0;
    if (p_inexact) {
        *p_inexact = inexact;
    }
    if (!inexact || trunc) {
        return rounded;
    }

    uint32_t lsb = rounded & 1;
    uint32_t inc = round_increment(rm, sign_negative, lsb, guard, sticky);
    return rounded + inc;
}

static uint32_t round_to_fp32(uint64_t mant_result, int32_t exp_result, uint32_t sign_result,
                              uint32_t *p_mxcsr) {
    uint32_t rm = (*p_mxcsr >> 13) & 3;
    if (!mant_result) {
        return (rm == 1) ? FP32_SIGN_MASK : 0; // exact zero; rm=1 gives -0
    }

    // Normalize to bit 53 always set. clz>=10 is guaranteed by every caller:
    //   fpemu_fp32_add/sub: mant_a, mant_b <= 2^52 (1.52 fixed-point); |sum| <= 2^53
    //     ordinarily, and the possible carry out of the add keeps |mant_result| < 2^54.
    //   fpemu_fp32_mul: (1.23 * 1.23) << 6 = 2.52 fixed-point; product <= (2^24-1)^2
    //     << 6, which is strictly less than 2^54.
    //   fpemu_fp32_div: (mant_a << 53) / mant_b with both mantissas in [2^23, 2^24)
    //     gives a quotient in (2^52, 2^54]; ORing the sticky bit at position 0 does not
    //     change the high bits. Worst case is mant_a = 2^24 - 1, mant_b = 2^23, which
    //     yields q = 2^54 - 2^30 < 2^54 (so clz is exactly 10, not smaller).
    //   fpemu_fp64_to_fp32: mant_a is a 53-bit FP64 mantissa, so clz is exactly 10 or 11.
    int clz = __builtin_clzll(mant_result);
    FPEMU_ASSERT(clz >= 10, "round_to_fp32 clz=%d mant_result=0x%016llx", clz,
                 (unsigned long long)mant_result);
    int shift0 = clz - 10;
    mant_result <<= shift0;
    exp_result += 1 - shift0;

    // Phase 1 -- round at NORMAL (24-bit) precision assuming unlimited exponent range. SSE
    // defines #U tininess as "the rounded result is less than min-normal when rounded to the
    // destination's precision with unlimited exponent range". If this phase carries the
    // mantissa into min-normal, the result is not tiny and no UE is raised even when the rounded
    // value is exactly min-normal.
    const int nshift = 30;
    uint32_t mant_n = (mant_result >> nshift) & FP32_FRAC_MASK;
    uint32_t guard_n = (mant_result >> (nshift - 1)) & 1;
    uint32_t sticky_n = (mant_result & ((1ull << (nshift - 1)) - 1)) != 0;
    bool inexact_n = (guard_n | sticky_n) != 0;
    int32_t exp_n = exp_result;
    uint32_t inc_n = round_increment(rm, sign_result, mant_n, guard_n, sticky_n);
    if (inc_n) {
        mant_n++;
        if (mant_n == FP32_IMPLICIT_BIT) {
            mant_n = 0;
            exp_n++;
        }
    }
    bool tiny = (exp_n <= 0);

    // Overflow: decided purely by the normal-range rounding.
    if (exp_n >= 255) {
        *p_mxcsr |= FPEMU_MXCSR_OE | FPEMU_MXCSR_PE;
        if (rm == 0) {
            return sign_result | FP32_INF;
        } else if (rm == 1) {
            return sign_result ? (sign_result | FP32_INF) : (sign_result | FP32_MAX_FINITE);
        } else if (rm == 2) {
            return sign_result ? (sign_result | FP32_MAX_FINITE) : (sign_result | FP32_INF);
        } else {
            return sign_result | FP32_MAX_FINITE; // toward zero: max finite
        }
    }

    // Not tiny -- normal result straight from phase 1.
    if (!tiny) {
        if (inexact_n) {
            *p_mxcsr |= FPEMU_MXCSR_PE;
        }
        return sign_result | ((uint32_t)exp_n << 23) | (mant_n & FP32_FRAC_MASK);
    }

    // Tiny. FTZ flushes to signed zero and always raises UE|PE.
    if (*p_mxcsr & FPEMU_MXCSR_FTZ) {
        *p_mxcsr |= FPEMU_MXCSR_UE | FPEMU_MXCSR_PE;
        return sign_result;
    }

    // Phase 2 -- round at subnormal precision. Use exp_result (post-normalize,
    // not the phase-1-bumped exp_n) so the shift reflects the original subnormal alignment.
    int sshift = 30 + (1 - exp_result);
    uint32_t mant_s, guard_s, sticky_s;
    if (sshift > 54) {
        // All bits below the keep window -- treat as all-sticky.
        mant_s = 0;
        guard_s = 0;
        sticky_s = mant_result != 0;
    } else {
        mant_s = (mant_result >> sshift) & FP32_FRAC_MASK;
        guard_s = (mant_result >> (sshift - 1)) & 1;
        sticky_s = (mant_result & ((1ull << (sshift - 1)) - 1)) != 0;
    }
    bool inexact_s = (guard_s | sticky_s) != 0;
    int32_t exp_s = 0;
    uint32_t inc_s = round_increment(rm, sign_result, mant_s, guard_s, sticky_s);
    if (inc_s) {
        mant_s++;
        if (mant_s == FP32_IMPLICIT_BIT) {
            mant_s = 0;
            exp_s = 1;
        }
    }
    if (inexact_s) {
        *p_mxcsr |= FPEMU_MXCSR_UE | FPEMU_MXCSR_PE;
    }
    if (exp_s >= 1) {
        return sign_result | ((uint32_t)exp_s << 23) | 0;
    }
    return sign_result | (mant_s & FP32_FRAC_MASK);
}

uint32_t fpemu_fp32_add(uint32_t a, uint32_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    uint32_t nan_result;
    if (quiet_binary_nan32(a, b, p_mxcsr, &nan_result)) {
        return nan_result;
    }
    // Raise the denormal-operand exception for any subnormal operand (DAZ off).
    // Do this BEFORE the inf short-circuit -- hardware raises #D on
    // add(subnormal, inf) even though the numeric result is inf.
    if (!(*p_mxcsr & FPEMU_MXCSR_DAZ) && (FP32_IS_SUBNORMAL(a) || FP32_IS_SUBNORMAL(b))) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    if (FP32_IS_INF(a)) {
        if (b == (a ^ FP32_SIGN_MASK)) { // inf of opposite sign
            *p_mxcsr |= FPEMU_MXCSR_IE;
            return FP32_DEFAULT_NAN;
        }
        return a;
    }
    if (FP32_IS_INF(b)) {
        return b;
    }
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if ((a & FP32_ABS_MASK) < FP32_IMPLICIT_BIT) {
            a &= FP32_SIGN_MASK;
        }
        if ((b & FP32_ABS_MASK) < FP32_IMPLICIT_BIT) {
            b &= FP32_SIGN_MASK;
        }
    }
    if (!a && !b) {
        return 0; // +0 + +0 = +0
    }
    if ((a == FP32_SIGN_MASK) && (b == FP32_SIGN_MASK)) {
        return FP32_SIGN_MASK; // -0 + -0 = -0
    }

    uint32_t exp_a = (a >> 23) & 255;
    uint32_t exp_b = (b >> 23) & 255;
    int64_t mant_a = a & FP32_FRAC_MASK;
    int64_t mant_b = b & FP32_FRAC_MASK;
    // Decode into aligned fixed-point significands. Subnormals use biased exponent 1 with no
    // implicit bit, which keeps alignment by biased-exponent difference valid.
    if (exp_a) {
        mant_a |= FP32_IMPLICIT_BIT;
    } else {
        exp_a = 1;
    }
    if (exp_b) {
        mant_b |= FP32_IMPLICIT_BIT;
    } else {
        exp_b = 1;
    }
    mant_a <<= 29; // 1.52
    mant_b <<= 29; // 1.52

    int32_t shift = exp_a - exp_b;
    int32_t exp_result = exp_a;
    if (shift > 0) {
        if (shift > 63) {
            mant_b = mant_b != 0; // only sticky survives
        } else {
            // Low 29 bits are zero after 1.52 alignment; sticky appears only when
            // alignment shifts past them.
            uint32_t sticky = (mant_b & ((1ull << shift) - 1)) != 0;
            mant_b = (mant_b >> shift) | sticky;
        }
    } else if (shift < 0) {
        shift = -shift;
        exp_result = exp_b;
        if (shift > 63) {
            mant_a = mant_a != 0; // only sticky survives
        } else {
            // Low 29 bits are zero after 1.52 alignment; sticky appears only when
            // alignment shifts past them.
            uint32_t sticky = (mant_a & ((1ull << shift) - 1)) != 0;
            mant_a = (mant_a >> shift) | sticky;
        }
    }
    if (a & FP32_SIGN_MASK) {
        mant_a = -mant_a;
    }
    if (b & FP32_SIGN_MASK) {
        mant_b = -mant_b;
    }

    int64_t mant_result = mant_a + mant_b;
    uint32_t sign_result = 0;
    if (mant_result < 0) {
        sign_result = FP32_SIGN_MASK;
        mant_result = -mant_result;
    }
    return round_to_fp32(mant_result, exp_result, sign_result, p_mxcsr);
}

uint32_t fpemu_fp32_sub(uint32_t a, uint32_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    // SUBSS propagates b's NaN without flipping its sign, so handle NaN explicitly here rather
    // than letting fpemu_fp32_add see a sign-flipped NaN that no longer matches the original
    // operand's payload.
    uint32_t nan_result;
    if (quiet_binary_nan32(a, b, p_mxcsr, &nan_result)) {
        return nan_result;
    }
    // Flip b's sign before delegating to add. Note: subnormal-detection and DE flag raising
    // happen inside fpemu_fp32_add; flipping a subnormal's sign keeps it subnormal.
    return fpemu_fp32_add(a, b ^ FP32_SIGN_MASK, p_mxcsr);
}

uint32_t fpemu_fp32_mul(uint32_t a, uint32_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    uint32_t nan_result;
    if (quiet_binary_nan32(a, b, p_mxcsr, &nan_result)) {
        return nan_result;
    }
    // Raise DE if any operand is a true subnormal and DAZ is off.
    if (!(*p_mxcsr & FPEMU_MXCSR_DAZ) && (FP32_IS_SUBNORMAL(a) || FP32_IS_SUBNORMAL(b))) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if ((a & FP32_ABS_MASK) < FP32_IMPLICIT_BIT) {
            a &= FP32_SIGN_MASK;
        }
        if ((b & FP32_ABS_MASK) < FP32_IMPLICIT_BIT) {
            b &= FP32_SIGN_MASK;
        }
    }
    uint32_t sign_result = (a ^ b) & FP32_SIGN_MASK;
    a &= FP32_ABS_MASK;
    b &= FP32_ABS_MASK;
    if (a == FP32_INF) {
        if (!b) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
            return FP32_DEFAULT_NAN;
        }
        return sign_result | FP32_INF;
    }
    if (b == FP32_INF) {
        if (!a) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
            return FP32_DEFAULT_NAN;
        }
        return sign_result | FP32_INF;
    }

    // Decode finite operands. Subnormal mantissas have no implicit bit but use biased exponent 1
    // so the product exponent formula still lines up with normal operands.
    uint32_t exp_a = a >> 23;
    uint32_t exp_b = b >> 23;
    uint32_t mant_a = a & FP32_FRAC_MASK;
    uint32_t mant_b = b & FP32_FRAC_MASK;
    if (exp_a) {
        mant_a |= FP32_IMPLICIT_BIT;
    } else if (!mant_a) {
        return sign_result;
    } else {
        exp_a = 1;
    }
    if (exp_b) {
        mant_b |= FP32_IMPLICIT_BIT;
    } else if (!mant_b) {
        return sign_result;
    } else {
        exp_b = 1;
    }
    uint64_t mant_result = (uint64_t)mant_a * (uint64_t)mant_b; // 1.23 * 1.23 = 2.46
    mant_result <<= 6; // 2.52 for round_to_fp32
    int32_t exp_result = exp_a + exp_b - 127;
    return round_to_fp32(mant_result, exp_result, sign_result, p_mxcsr);
}

static uint32_t fpemu_fp32_minmax(uint32_t a, uint32_t b, bool is_max, uint32_t *p_mxcsr) {
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP32_IS_SUBNORMAL(a)) {
            a &= FP32_SIGN_MASK;
        }
        if (FP32_IS_SUBNORMAL(b)) {
            b &= FP32_SIGN_MASK;
        }
    }
    if (FP32_IS_NAN(a) || FP32_IS_NAN(b)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return b;
    }
    if (!(*p_mxcsr & FPEMU_MXCSR_DAZ) && (FP32_IS_SUBNORMAL(a) || FP32_IS_SUBNORMAL(b))) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    bool a_less = fp32_ordered_less(a, b);
    if (is_max) {
        bool b_less = fp32_ordered_less(b, a);
        return b_less ? a : b;
    }
    return a_less ? a : b;
}

uint32_t fpemu_fp32_min(uint32_t a, uint32_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_minmax(a, b, false, p_mxcsr);
}

uint32_t fpemu_fp32_max(uint32_t a, uint32_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_minmax(a, b, true, p_mxcsr);
}

uint32_t fpemu_fp32_round(uint32_t a, uint32_t imm, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    FPEMU_ASSERT(imm <= 15, "ROUNDSS imm out of range: imm=0x%x", imm);
    if (FP32_IS_NAN(a)) {
        if (!(a & FP32_QUIET_MASK)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        return a | FP32_QUIET_MASK;
    }
    if (FP32_IS_INF(a) || !(a & FP32_ABS_MASK)) {
        return a;
    }
    uint32_t sign = a & FP32_SIGN_MASK;
    uint32_t abs_a = a & FP32_ABS_MASK;
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP32_IS_SUBNORMAL(a)) {
            return sign;
        }
    }

    uint32_t rm = round_control_from_imm(imm, *p_mxcsr);
    bool suppress_precision = (imm & 8) != 0;
    uint32_t exp = (abs_a >> 23) & 0xFFu;
    int32_t e = exp ? ((int32_t)exp - 127) : -126;
    uint32_t rounded_abs = abs_a;
    bool inexact = false;

    if (e < 0) {
        inexact = true;
        bool increment = false;
        if (rm == 0) {
            increment = abs_a > 0x3F000000u;
        } else if (rm == 1) {
            increment = sign != 0;
        } else if (rm == 2) {
            increment = sign == 0;
        }
        rounded_abs = increment ? 0x3F800000u : 0;
    } else if (e < 23) {
        uint32_t frac_bits = 23u - (uint32_t)e;
        uint32_t mask = (1u << frac_bits) - 1u;
        uint32_t dropped = abs_a & mask;
        if (dropped != 0) {
            inexact = true;
            uint32_t kept = abs_a & ~mask;
            uint32_t guard = (dropped >> (frac_bits - 1)) & 1u;
            uint32_t sticky =
                (frac_bits > 1) ? ((dropped & ((1u << (frac_bits - 1)) - 1u)) != 0) : 0;
            uint32_t lsb = (kept >> frac_bits) & 1u;
            uint32_t inc = round_increment(rm, sign != 0, lsb, guard, sticky);
            rounded_abs = kept + (inc ? (1u << frac_bits) : 0);
        }
    }

    if (inexact && !suppress_precision) {
        *p_mxcsr |= FPEMU_MXCSR_PE;
    }
    return sign | rounded_abs;
}

uint32_t fpemu_fp32_cmp(uint32_t a, uint32_t b, uint32_t imm, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    FPEMU_ASSERT(imm <= 31, "CMPSS imm out of range: imm=0x%x", imm);
    bool unordered = FP32_IS_NAN(a) || FP32_IS_NAN(b);
    if (unordered) {
        if (FP32_IS_SNAN(a) || FP32_IS_SNAN(b) || cmp_predicate_signals(imm)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        return cmp_predicate_value(imm, true, false, false) ? 0xFFFFFFFFu : 0;
    }
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP32_IS_SUBNORMAL(a)) {
            a &= FP32_SIGN_MASK;
        }
        if (FP32_IS_SUBNORMAL(b)) {
            b &= FP32_SIGN_MASK;
        }
    } else if (FP32_IS_SUBNORMAL(a) || FP32_IS_SUBNORMAL(b)) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    bool less = fp32_ordered_less(a, b);
    bool equal = fp32_ordered_equal(a, b);
    return cmp_predicate_value(imm, false, less, equal) ? 0xFFFFFFFFu : 0;
}

static uint32_t fpemu_fp32_comi_core(uint32_t a, uint32_t b, bool quiet, uint32_t *p_mxcsr) {
    bool unordered = FP32_IS_NAN(a) || FP32_IS_NAN(b);
    if (unordered) {
        if (!quiet || FP32_IS_SNAN(a) || FP32_IS_SNAN(b)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        return comi_flags(true, false, false);
    }
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP32_IS_SUBNORMAL(a)) {
            a &= FP32_SIGN_MASK;
        }
        if (FP32_IS_SUBNORMAL(b)) {
            b &= FP32_SIGN_MASK;
        }
    } else if (FP32_IS_SUBNORMAL(a) || FP32_IS_SUBNORMAL(b)) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    bool less = fp32_ordered_less(a, b);
    bool equal = fp32_ordered_equal(a, b);
    return comi_flags(false, less, equal);
}

uint32_t fpemu_fp32_comi(uint32_t a, uint32_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_comi_core(a, b, false, p_mxcsr);
}

uint32_t fpemu_fp32_ucomi(uint32_t a, uint32_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_comi_core(a, b, true, p_mxcsr);
}

static uint32_t fma_default_nan32(uint32_t *p_mxcsr) {
    *p_mxcsr |= FPEMU_MXCSR_IE;
    return FP32_DEFAULT_NAN;
}

static uint64_t fma_default_nan64(uint32_t *p_mxcsr) {
    *p_mxcsr |= FPEMU_MXCSR_IE;
    return FP64_DEFAULT_NAN;
}

static void decode_finite_fp32(uint32_t x, uint64_t *p_mant, int32_t *p_exp) {
    uint32_t raw_exp = (x >> 23) & 0xFF;
    uint32_t frac = x & FP32_FRAC_MASK;
    if (raw_exp != 0) {
        *p_mant = (uint64_t)FP32_IMPLICIT_BIT | frac;
        *p_exp = (int32_t)raw_exp - 127 - 23;
    } else {
        *p_mant = frac;
        *p_exp = 1 - 127 - 23;
    }
}

static void decode_finite_fp64(uint64_t x, uint64_t *p_mant, int32_t *p_exp) {
    uint32_t raw_exp = (uint32_t)((x >> 52) & 0x7FF);
    uint64_t frac = x & FP64_FRAC_MASK;
    if (raw_exp != 0) {
        *p_mant = FP64_IMPLICIT_BIT | frac;
        *p_exp = (int32_t)raw_exp - 1023 - 52;
    } else {
        *p_mant = frac;
        *p_exp = 1 - 1023 - 52;
    }
}

static __uint128_t fma_window_term_u128(__uint128_t mant, int32_t exp, int32_t base) {
    if (!mant) {
        return 0;
    }
    if (exp >= base) {
        int shift = exp - base;
        FPEMU_ASSERT(shift < 128, "fma_window_term_u128 lshift=%d", shift);
        return mant << shift;
    }
    int shift = base - exp;
    if (shift >= 128) {
        return 1;
    }
    __uint128_t dropped_mask = (((__uint128_t)1) << shift) - 1;
    __uint128_t dropped = mant & dropped_mask;
    mant >>= shift;
    if (dropped != 0) {
        mant |= 1;
    }
    return mant;
}

static void fma_signed_add_i128(__int128_t *p_acc, __uint128_t mag, bool neg) {
    FPEMU_ASSERT(mag < (((__uint128_t)1) << 127), "fma_signed_add_i128 mag overflow");
    if (neg) {
        *p_acc -= (__int128_t)mag;
    } else {
        *p_acc += (__int128_t)mag;
    }
}

static uint32_t round_i128_to_fp32(__int128_t acc, int32_t base, uint32_t *p_mxcsr) {
    bool neg = acc < 0;
    __uint128_t mag = neg ? (__uint128_t)(-acc) : (__uint128_t)acc;
    if (!mag) {
        return round_to_fp32(0, 0, 0, p_mxcsr);
    }
    int top = 127 - clz_u128(mag);
    int shift = top - 53;
    uint64_t mant;
    int32_t out_base;
    if (shift > 0) {
        __uint128_t dropped_mask = (((__uint128_t)1) << shift) - 1;
        __uint128_t dropped = mag & dropped_mask;
        mag >>= shift;
        if (dropped != 0) {
            mag |= 1;
        }
        mant = (uint64_t)mag;
        out_base = base + shift;
    } else {
        mant = (uint64_t)(mag << -shift);
        out_base = base + shift;
    }
    return round_to_fp32(mant, out_base + 127 + 52, neg ? FP32_SIGN_MASK : 0, p_mxcsr);
}

static uint64_t round_i128_to_fp64(__int128_t acc, int32_t base, uint32_t *p_mxcsr) {
    bool neg = acc < 0;
    __uint128_t mag = neg ? (__uint128_t)(-acc) : (__uint128_t)acc;
    if (!mag) {
        return round_to_fp64(0, 0, 0, p_mxcsr);
    }
    int top = 127 - clz_u128(mag);
    int shift = top - 105;
    __uint128_t mant;
    int32_t out_base;
    if (shift > 0) {
        __uint128_t dropped_mask = (((__uint128_t)1) << shift) - 1;
        __uint128_t dropped = mag & dropped_mask;
        mant = mag >> shift;
        if (dropped != 0) {
            mant |= 1;
        }
        out_base = base + shift;
    } else {
        mant = mag << -shift;
        out_base = base + shift;
    }
    return round_to_fp64(mant, out_base + 1023 + 104, neg ? FP64_SIGN_MASK : 0, p_mxcsr);
}

static uint32_t fpemu_fp32_fma_core(uint32_t a, uint32_t b, uint32_t c, bool neg_product,
                                    bool sub_c, uint32_t *p_mxcsr) {
    if (FP32_IS_NAN(a) || FP32_IS_NAN(b) || FP32_IS_NAN(c)) {
        if (FP32_IS_SNAN(a) || FP32_IS_SNAN(b) || FP32_IS_SNAN(c)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        if (FP32_IS_NAN(a)) {
            return a | FP32_QUIET_MASK;
        }
        if (FP32_IS_NAN(b)) {
            return b | FP32_QUIET_MASK;
        }
        return c | FP32_QUIET_MASK;
    }
    bool raw_sub = !(*p_mxcsr & FPEMU_MXCSR_DAZ) &&
                   (FP32_IS_SUBNORMAL(a) || FP32_IS_SUBNORMAL(b) || FP32_IS_SUBNORMAL(c));
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP32_IS_SUBNORMAL(a)) {
            a &= FP32_SIGN_MASK;
        }
        if (FP32_IS_SUBNORMAL(b)) {
            b &= FP32_SIGN_MASK;
        }
        if (FP32_IS_SUBNORMAL(c)) {
            c &= FP32_SIGN_MASK;
        }
    }
    bool invalid_mul = (FP32_IS_INF(a) && !(b & FP32_ABS_MASK)) ||
                       (FP32_IS_INF(b) && !(a & FP32_ABS_MASK));
    if (invalid_mul) {
        return fma_default_nan32(p_mxcsr);
    }

    uint32_t product_sign = ((a ^ b) & FP32_SIGN_MASK) ^ (neg_product ? FP32_SIGN_MASK : 0);
    uint32_t c_eff = c ^ (sub_c ? FP32_SIGN_MASK : 0);
    if (FP32_IS_INF(a) || FP32_IS_INF(b)) {
        uint32_t prod_inf = product_sign | FP32_INF;
        if (FP32_IS_INF(c_eff) && ((prod_inf ^ c_eff) == FP32_SIGN_MASK)) {
            return fma_default_nan32(p_mxcsr);
        }
        if (raw_sub) {
            *p_mxcsr |= FPEMU_MXCSR_DE;
        }
        return prod_inf;
    }
    if (FP32_IS_INF(c_eff)) {
        if (raw_sub) {
            *p_mxcsr |= FPEMU_MXCSR_DE;
        }
        return c_eff;
    }
    if (raw_sub) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }

    bool product_zero = !(a & FP32_ABS_MASK) || !(b & FP32_ABS_MASK);
    bool c_zero = !(c_eff & FP32_ABS_MASK);
    if (product_zero && c_zero) {
        uint32_t rm = (*p_mxcsr >> 13) & 3;
        if (product_sign == (c_eff & FP32_SIGN_MASK)) {
            return product_sign;
        }
        return (rm == 1) ? FP32_SIGN_MASK : 0;
    }

    uint64_t mant_a, mant_b, mant_c;
    int32_t exp_a, exp_b, exp_c;
    decode_finite_fp32(a & FP32_ABS_MASK, &mant_a, &exp_a);
    decode_finite_fp32(b & FP32_ABS_MASK, &mant_b, &exp_b);
    decode_finite_fp32(c_eff & FP32_ABS_MASK, &mant_c, &exp_c);

    __uint128_t product = (__uint128_t)mant_a * mant_b;
    int32_t exp_product = exp_a + exp_b;
    int product_top = (product != 0) ? exp_product + 127 - clz_u128(product) : -1000000;
    int c_top = (mant_c != 0) ? exp_c + (63 - __builtin_clzll(mant_c)) : -1000000;
    int32_t low = exp_product < exp_c ? exp_product : exp_c;
    int32_t high = product_top > c_top ? product_top : c_top;

    int32_t base;
    // Exact path: align both terms without exceeding signed __int128 range. Far path:
    // cancellation is impossible, so bits below the retained window only affect sticky.
    if (high - low <= 126) {
        if (!product) {
            base = exp_c;
        } else if (!mant_c) {
            base = exp_product;
        } else {
            base = low;
        }
    } else {
        base = high - 100;
    }

    __int128_t acc = 0;
    __uint128_t product_term = fma_window_term_u128(product, exp_product, base);
    __uint128_t c_term = fma_window_term_u128(mant_c, exp_c, base);
    fma_signed_add_i128(&acc, product_term, product_sign != 0);
    fma_signed_add_i128(&acc, c_term, (c_eff & FP32_SIGN_MASK) != 0);
    return round_i128_to_fp32(acc, base, p_mxcsr);
}

uint32_t fpemu_fp32_fmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_fma_core(a, b, c, false, false, p_mxcsr);
}

uint32_t fpemu_fp32_fmsub(uint32_t a, uint32_t b, uint32_t c, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_fma_core(a, b, c, false, true, p_mxcsr);
}

uint32_t fpemu_fp32_fnmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_fma_core(a, b, c, true, false, p_mxcsr);
}

uint32_t fpemu_fp32_fnmsub(uint32_t a, uint32_t b, uint32_t c, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_fma_core(a, b, c, true, true, p_mxcsr);
}

uint32_t fpemu_fp32_div(uint32_t a, uint32_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    // NaN short-circuit. DIVSS follows the same "a wins" convention as ADD/SUB/MUL: NaN
    // suppresses DE even if the other operand is subnormal, and only #I is raised (by an sNaN
    // anywhere). qNaN with payload is propagated with the quiet bit OR'd in and the sign preserved.
    uint32_t nan_result;
    if (quiet_binary_nan32(a, b, p_mxcsr, &nan_result)) {
        return nan_result;
    }

    // Capture whether DE would fire based on the *original* operands and DAZ. This is only
    // consulted on non-exceptional paths -- DIVSS suppresses #D when the operation raises #I
    // (0/0 or inf/inf) or #Z (finite_nonzero / 0). Notable hardware probe: min_sub / +0 raises
    // only #Z, not #D, even though the numerator is subnormal.
    bool raw_sub = !(*p_mxcsr & FPEMU_MXCSR_DAZ) && (FP32_IS_SUBNORMAL(a) || FP32_IS_SUBNORMAL(b));

    // inf / inf -> #I, NaN. DAZ does not touch infinities so handle this pre-flush.
    if (FP32_IS_INF(a) && FP32_IS_INF(b)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return FP32_DEFAULT_NAN;
    }

    // DAZ: flush subnormal operands to signed zero before the remaining checks.
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP32_IS_SUBNORMAL(a)) {
            a &= FP32_SIGN_MASK;
        }
        if (FP32_IS_SUBNORMAL(b)) {
            b &= FP32_SIGN_MASK;
        }
    }

    // 0 / 0 (post-DAZ). Catches native 0/0 and DAZ-induced subnormal/subnormal or
    // subnormal/zero collapses. raw_sub is already false under DAZ so #D is suppressed.
    if (!(a & FP32_ABS_MASK) && !(b & FP32_ABS_MASK)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return FP32_DEFAULT_NAN;
    }

    uint32_t sign_result = (a ^ b) & FP32_SIGN_MASK;

    // inf / finite_nonzero -> signed inf. #D fires if divisor was subnormal
    // (raw_sub captures that; #Z is not raised because divisor is nonzero).
    if (FP32_IS_INF(a)) {
        if (raw_sub) {
            *p_mxcsr |= FPEMU_MXCSR_DE;
        }
        return sign_result | FP32_INF;
    }

    // finite_nonzero / 0 -> #Z, signed inf. No #D even when numerator was subnormal: hardware
    // probe min_sub/+0 raises only #Z.
    if (!(b & FP32_ABS_MASK)) {
        *p_mxcsr |= FPEMU_MXCSR_ZE;
        return sign_result | FP32_INF;
    }

    // Remaining paths all see "regular" subnormals through to #D, if applicable.
    if (raw_sub) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }

    // 0 / finite_nonzero -> signed zero. Divisor may be subnormal (handled by raw_sub above) or inf
    // (no extra flag).
    if (!(a & FP32_ABS_MASK)) {
        return sign_result;
    }

    // finite_nonzero / inf -> signed zero. No extra flag.
    if (FP32_IS_INF(b)) {
        return sign_result;
    }

    // Normal division: both finite, nonzero; may be subnormal only if DAZ is off.
    int32_t exp_a = (int32_t)((a >> 23) & 255);
    int32_t exp_b = (int32_t)((b >> 23) & 255);
    uint32_t mant_a = a & FP32_FRAC_MASK;
    uint32_t mant_b = b & FP32_FRAC_MASK;
    if (exp_a) {
        mant_a |= FP32_IMPLICIT_BIT;
    } else {
        // Subnormal: shift until bit 23 is set. Adjust the biased exponent downward by the same
        // amount so the effective value is preserved.
        // exp_a can go negative here; round_to_fp32 clips or adjusts the final exponent.
        int shift = __builtin_clz(mant_a) - 8; // leaves bit 23 as MSB
        mant_a <<= shift;
        exp_a = 1 - shift;
    }
    if (exp_b) {
        mant_b |= FP32_IMPLICIT_BIT;
    } else {
        int shift = __builtin_clz(mant_b) - 8;
        mant_b <<= shift;
        exp_b = 1 - shift;
    }

    // Quotient layout. With mant_a, mant_b both in [2^23, 2^24), the exact ratio lies in
    // (0.5, 2). Compute q = (mant_a << 53) / mant_b; then q is in (2^52, 2^54]. Bit 53 is
    // set when mant_a >= mant_b, else bit 52 is the top. round_to_fp32 expects a value with
    // clz >= 10 and renormalizes via its own shift0, so both cases feed in uniformly. The
    // remainder of the division supplies the sticky bit -- ORing it into bit 0 is safe because
    // round_to_fp32's phase-1 sticky_n covers bits [28..0] and phase-2 treats the whole
    // register below its keep window as sticky.
    __uint128_t num128 = (__uint128_t)mant_a << 53;
    uint64_t q = (uint64_t)(num128 / mant_b);
    uint64_t rem = (uint64_t)(num128 % mant_b);
    uint64_t mant_result = q | (rem ? 1ull : 0ull);

    // Exponent derivation: with ratio in (0.5, 2) and biased exps ea, eb, the final biased
    // exponent is either (ea - eb + 127) when ratio >= 1 or (ea - eb + 126) when ratio < 1.
    // round_to_fp32 adds (1 - shift0) which exactly compensates: shift0=0 for ratio>=1 case,
    // shift0=1 for ratio<1 case. Passing (ea - eb + 126) works for both.
    int32_t exp_result = exp_a - exp_b + 126;
    return round_to_fp32(mant_result, exp_result, sign_result, p_mxcsr);
}

// Bit-by-bit integer square root. Returns floor(sqrt(n)); *p_rem receives n - res^2 so callers
// can distinguish an exact square (rem == 0) from an inexact one. 128-bit input is enough for
// our sqrt scaling (n < 2^106), and 64-bit return fits res < 2^53. Callers must ensure n != 0;
// the initial `while (bit > n)` loop would spin otherwise.
static uint64_t isqrt128(__uint128_t n, __uint128_t *p_rem) {
    __uint128_t res = 0;
    __uint128_t bit = ((__uint128_t)1) << 126; // highest power of 4 representable in 128b
    while (bit > n) {
        bit >>= 2;
    }
    while (bit != 0) {
        __uint128_t t = res + bit;
        res >>= 1;
        if (n >= t) {
            n -= t;
            res += bit;
        }
        bit >>= 2;
    }
    *p_rem = n;
    return (uint64_t)res;
}

uint32_t fpemu_fp32_sqrt(uint32_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    // NaN short-circuit. SQRTSS mirrors the add/sub/mul/div convention: NaN suppresses #D and
    // #I-from-sign; only an sNaN raises #I. qNaN with payload is propagated verbatim (quiet bit
    // already set). Hardware probe confirms #D is NOT raised even when the NaN's low mantissa
    // bits happen to look like a subnormal encoding.
    if (FP32_IS_NAN(a)) {
        if (!(a & FP32_QUIET_MASK)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        return a | FP32_QUIET_MASK;
    }
    // DAZ: subnormal operand pre-flushes to signed zero. This happens before the negative-input
    // -> #I decision, which is why sqrt(-subnormal) under DAZ is -0 (sqrt(-0) rule) with no
    // flag, while DAZ off lands on the #I path.
    if ((*p_mxcsr & FPEMU_MXCSR_DAZ) && FP32_IS_SUBNORMAL(a)) {
        a &= FP32_SIGN_MASK;
    }
    // sqrt(+/-0) = signed zero, no flag. sqrt(+inf) = +inf, no flag.
    if (!(a & FP32_ABS_MASK)) {
        return a;
    }
    if (a == FP32_INF) {
        return a;
    }
    // Negative finite (or -inf): #I only. Hardware probe on sqrt(-min_sub) with
    // DAZ off confirms only #I fires -- #D is suppressed even with a subnormal negative operand.
    if (a & FP32_SIGN_MASK) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return FP32_DEFAULT_NAN;
    }
    // Positive, finite, nonzero. Subnormal raises #D (DAZ off path).
    if (FP32_IS_SUBNORMAL(a)) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }

    int32_t raw_exp = (a >> 23) & 0xFF;
    uint32_t mant_bits = a & FP32_FRAC_MASK;
    uint32_t mant; // 24-bit normalized mantissa, MSB at bit 23
    int32_t unbiased_exp; // value = mant * 2^(unbiased_exp - 23)
    if (raw_exp) {
        mant = mant_bits | FP32_IMPLICIT_BIT;
        unbiased_exp = raw_exp - 127;
    } else {
        // Subnormal: left-shift mantissa until bit 23 is MSB, decrementing the implicit
        // exponent by the same amount. For FP32, mant_bits is a 23-bit value, so __builtin_clz
        // on a uint32_t gives a result in [9, 31]; subtract 8 to land bit 23 exactly on the MSB.
        int shift = __builtin_clz(mant_bits) - 8;
        mant = mant_bits << shift;
        unbiased_exp = 1 - shift - 127;
    }

    // Scale into a fixed-point value with value = scaled * 2^delta, delta even, and scaled in
    // [2^52, 2^54). Parity of unbiased_exp picks the shift so delta is even: for even
    // unbiased_exp, unbiased_exp - 23 is odd, so shift mant by 29 to absorb one factor
    // (scaled MSB at bit 52, delta = unbiased_exp - 52). For odd unbiased_exp, shift mant by 30
    // (scaled MSB at bit 53, delta = unbiased_exp - 53). In both cases isqrt(scaled << 52)
    // returns root with MSB at bit 52: sqrt(scaled) * 2^26 lands in [2^52, 2^53).
    uint64_t scaled = (!(unbiased_exp & 1)) ? ((uint64_t)mant << 29) : ((uint64_t)mant << 30);
    __uint128_t rem;
    uint64_t root = isqrt128((__uint128_t)scaled << 52, &rem);

    // Feed round_to_fp32. Shift root up by 1 so its MSB sits at bit 53 (round_to_fp32's expected
    // implicit-bit position), then OR a single sticky bit at bit 0 to represent "sqrt was inexact"
    // (rem != 0). Phase-1 sticky_n covers bits [28:0], so the OR is seen by the rounder without
    // disturbing mantissa or guard bits.
    uint64_t mant_result = (root << 1) | (rem ? 1ull : 0ull);
    // Biased result exponent = floor(unbiased_exp/2) + 127. Avoid relying on sign-preserving
    // right shift of a signed int by doing the split explicitly.
    int32_t e_r_unbiased = (unbiased_exp & 1) ? ((unbiased_exp - 1) / 2) : (unbiased_exp / 2);
    int32_t exp_result_in = e_r_unbiased + 127 - 1; // undo round_to_fp32's +1
    uint32_t sign_result = 0;
    return round_to_fp32(mant_result, exp_result_in, sign_result, p_mxcsr);
}

uint64_t fpemu_fp64_add(uint64_t a, uint64_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    uint64_t nan_result;
    if (quiet_binary_nan64(a, b, p_mxcsr, &nan_result)) {
        return nan_result;
    }
    // Raise the denormal-operand exception for any subnormal operand unless DAZ masks it out.
    if (!(*p_mxcsr & FPEMU_MXCSR_DAZ) && (FP64_IS_SUBNORMAL(a) || FP64_IS_SUBNORMAL(b))) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP64_IS_SUBNORMAL(a)) {
            a &= FP64_SIGN_MASK;
        }
        if (FP64_IS_SUBNORMAL(b)) {
            b &= FP64_SIGN_MASK;
        }
    }
    if (!a && !b) {
        return 0; // +0 + +0 = +0
    }
    if ((a == FP64_SIGN_MASK) && (b == FP64_SIGN_MASK)) {
        return FP64_SIGN_MASK; // -0 + -0 = -0
    }

    if (FP64_IS_INF(a)) {
        if (b == (a ^ FP64_SIGN_MASK)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
            return FP64_DEFAULT_NAN;
        }
        return a;
    }
    if (FP64_IS_INF(b)) {
        return b;
    }

    uint64_t exp_a = (a >> 52) & 2047;
    uint64_t exp_b = (b >> 52) & 2047;
    __uint128_t mant_a = a & FP64_FRAC_MASK;
    __uint128_t mant_b = b & FP64_FRAC_MASK;
    if (exp_a) {
        mant_a |= FP64_IMPLICIT_BIT;
    } else {
        exp_a = 1;
    }
    if (exp_b) {
        mant_b |= FP64_IMPLICIT_BIT;
    } else {
        exp_b = 1;
    }
    mant_a <<= 52; // 1.105
    mant_b <<= 52; // 1.105

    int32_t shift = (int32_t)exp_a - (int32_t)exp_b;
    int32_t exp_result = (int32_t)exp_a;
    if (shift > 0) {
        mant_b = align_right_with_sticky_u128(mant_b, shift);
    } else if (shift < 0) {
        shift = -shift;
        exp_result = (int32_t)exp_b;
        mant_a = align_right_with_sticky_u128(mant_a, shift);
    }
    __int128_t signed_mant_a = (__int128_t)mant_a;
    __int128_t signed_mant_b = (__int128_t)mant_b;
    if (a & FP64_SIGN_MASK) {
        signed_mant_a = -signed_mant_a;
    }
    if (b & FP64_SIGN_MASK) {
        signed_mant_b = -signed_mant_b;
    }

    __int128_t mant_result = signed_mant_a + signed_mant_b;
    uint64_t sign_result = 0;
    if (mant_result < 0) {
        sign_result = FP64_SIGN_MASK;
        mant_result = -mant_result;
    }
    return round_to_fp64((__uint128_t)mant_result, exp_result, sign_result, p_mxcsr);
}

uint64_t fpemu_fp64_sub(uint64_t a, uint64_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    uint64_t nan_result;
    if (quiet_binary_nan64(a, b, p_mxcsr, &nan_result)) {
        return nan_result;
    }
    // SUBSD propagates b's NaN without flipping its sign, so only non-NaN path
    // delegates to add with flipped `b` sign.
    return fpemu_fp64_add(a, b ^ FP64_SIGN_MASK, p_mxcsr);
}

uint64_t fpemu_fp64_mul(uint64_t a, uint64_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    uint64_t nan_result;
    if (quiet_binary_nan64(a, b, p_mxcsr, &nan_result)) {
        return nan_result;
    }
    if (!(*p_mxcsr & FPEMU_MXCSR_DAZ) && (FP64_IS_SUBNORMAL(a) || FP64_IS_SUBNORMAL(b))) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP64_IS_SUBNORMAL(a)) {
            a &= FP64_SIGN_MASK;
        }
        if (FP64_IS_SUBNORMAL(b)) {
            b &= FP64_SIGN_MASK;
        }
    }

    uint64_t sign_result = (a ^ b) & FP64_SIGN_MASK;
    a &= FP64_ABS_MASK;
    b &= FP64_ABS_MASK;
    if (a == FP64_INF) {
        if (!b) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
            return FP64_DEFAULT_NAN;
        }
        return sign_result | FP64_INF;
    }
    if (b == FP64_INF) {
        if (!a) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
            return FP64_DEFAULT_NAN;
        }
        return sign_result | FP64_INF;
    }

    uint64_t exp_a = a >> 52;
    uint64_t exp_b = b >> 52;
    uint64_t mant_a = a & FP64_FRAC_MASK;
    uint64_t mant_b = b & FP64_FRAC_MASK;
    if (exp_a) {
        mant_a |= FP64_IMPLICIT_BIT;
    } else if (!mant_a) {
        return sign_result;
    } else {
        exp_a = 1;
    }
    if (exp_b) {
        mant_b |= FP64_IMPLICIT_BIT;
    } else if (!mant_b) {
        return sign_result;
    } else {
        exp_b = 1;
    }

    __uint128_t mant_result = (__uint128_t)mant_a * (__uint128_t)mant_b;
    int32_t exp_result = (int32_t)exp_a + (int32_t)exp_b - 1023;
    return round_to_fp64(mant_result, exp_result, sign_result, p_mxcsr);
}

static uint64_t fpemu_fp64_minmax(uint64_t a, uint64_t b, bool is_max, uint32_t *p_mxcsr) {
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP64_IS_SUBNORMAL(a)) {
            a &= FP64_SIGN_MASK;
        }
        if (FP64_IS_SUBNORMAL(b)) {
            b &= FP64_SIGN_MASK;
        }
    }
    if (FP64_IS_NAN(a) || FP64_IS_NAN(b)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return b;
    }
    if (!(*p_mxcsr & FPEMU_MXCSR_DAZ) && (FP64_IS_SUBNORMAL(a) || FP64_IS_SUBNORMAL(b))) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    bool a_less = fp64_ordered_less(a, b);
    if (is_max) {
        bool b_less = fp64_ordered_less(b, a);
        return b_less ? a : b;
    }
    return a_less ? a : b;
}

uint64_t fpemu_fp64_min(uint64_t a, uint64_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_minmax(a, b, false, p_mxcsr);
}

uint64_t fpemu_fp64_max(uint64_t a, uint64_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_minmax(a, b, true, p_mxcsr);
}

uint64_t fpemu_fp64_round(uint64_t a, uint32_t imm, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    FPEMU_ASSERT(imm <= 15, "ROUNDSD imm out of range: imm=0x%x", imm);
    if (FP64_IS_NAN(a)) {
        if (!(a & FP64_QUIET_MASK)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        return a | FP64_QUIET_MASK;
    }
    if (FP64_IS_INF(a) || !(a & FP64_ABS_MASK)) {
        return a;
    }
    uint64_t sign = a & FP64_SIGN_MASK;
    uint64_t abs_a = a & FP64_ABS_MASK;
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP64_IS_SUBNORMAL(a)) {
            return sign;
        }
    }

    uint32_t rm = round_control_from_imm(imm, *p_mxcsr);
    bool suppress_precision = (imm & 8) != 0;
    uint32_t exp = (uint32_t)((abs_a >> 52) & 0x7FFu);
    int32_t e = exp ? ((int32_t)exp - 1023) : -1022;
    uint64_t rounded_abs = abs_a;
    bool inexact = false;

    if (e < 0) {
        inexact = true;
        bool increment = false;
        if (rm == 0) {
            increment = abs_a > 0x3FE0000000000000ull;
        } else if (rm == 1) {
            increment = sign != 0;
        } else if (rm == 2) {
            increment = sign == 0;
        }
        rounded_abs = increment ? 0x3FF0000000000000ull : 0;
    } else if (e < 52) {
        uint32_t frac_bits = 52u - (uint32_t)e;
        uint64_t mask = (1ull << frac_bits) - 1ull;
        uint64_t dropped = abs_a & mask;
        if (dropped != 0) {
            inexact = true;
            uint64_t kept = abs_a & ~mask;
            uint32_t guard = (uint32_t)((dropped >> (frac_bits - 1)) & 1ull);
            uint32_t sticky =
                (frac_bits > 1) ? ((dropped & ((1ull << (frac_bits - 1)) - 1ull)) != 0) : 0;
            uint32_t lsb = (uint32_t)((kept >> frac_bits) & 1ull);
            uint32_t inc = round_increment(rm, sign != 0, lsb, guard, sticky);
            rounded_abs = kept + (inc ? (1ull << frac_bits) : 0);
        }
    }

    if (inexact && !suppress_precision) {
        *p_mxcsr |= FPEMU_MXCSR_PE;
    }
    return sign | rounded_abs;
}

uint64_t fpemu_fp64_cmp(uint64_t a, uint64_t b, uint32_t imm, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    FPEMU_ASSERT(imm <= 31, "CMPSD imm out of range: imm=0x%x", imm);
    bool unordered = FP64_IS_NAN(a) || FP64_IS_NAN(b);
    if (unordered) {
        if (FP64_IS_SNAN(a) || FP64_IS_SNAN(b) || cmp_predicate_signals(imm)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        return cmp_predicate_value(imm, true, false, false) ? 0xFFFFFFFFFFFFFFFFull : 0;
    }
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP64_IS_SUBNORMAL(a)) {
            a &= FP64_SIGN_MASK;
        }
        if (FP64_IS_SUBNORMAL(b)) {
            b &= FP64_SIGN_MASK;
        }
    } else if (FP64_IS_SUBNORMAL(a) || FP64_IS_SUBNORMAL(b)) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    bool less = fp64_ordered_less(a, b);
    bool equal = fp64_ordered_equal(a, b);
    return cmp_predicate_value(imm, false, less, equal) ? 0xFFFFFFFFFFFFFFFFull : 0;
}

static uint32_t fpemu_fp64_comi_core(uint64_t a, uint64_t b, bool quiet, uint32_t *p_mxcsr) {
    bool unordered = FP64_IS_NAN(a) || FP64_IS_NAN(b);
    if (unordered) {
        if (!quiet || FP64_IS_SNAN(a) || FP64_IS_SNAN(b)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        return comi_flags(true, false, false);
    }
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP64_IS_SUBNORMAL(a)) {
            a &= FP64_SIGN_MASK;
        }
        if (FP64_IS_SUBNORMAL(b)) {
            b &= FP64_SIGN_MASK;
        }
    } else if (FP64_IS_SUBNORMAL(a) || FP64_IS_SUBNORMAL(b)) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    bool less = fp64_ordered_less(a, b);
    bool equal = fp64_ordered_equal(a, b);
    return comi_flags(false, less, equal);
}

uint32_t fpemu_fp64_comi(uint64_t a, uint64_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_comi_core(a, b, false, p_mxcsr);
}

uint32_t fpemu_fp64_ucomi(uint64_t a, uint64_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_comi_core(a, b, true, p_mxcsr);
}

static uint64_t fpemu_fp64_fma_core(uint64_t a, uint64_t b, uint64_t c, bool neg_product,
                                    bool sub_c, uint32_t *p_mxcsr) {
    if (FP64_IS_NAN(a) || FP64_IS_NAN(b) || FP64_IS_NAN(c)) {
        if (FP64_IS_SNAN(a) || FP64_IS_SNAN(b) || FP64_IS_SNAN(c)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        if (FP64_IS_NAN(a)) {
            return a | FP64_QUIET_MASK;
        }
        if (FP64_IS_NAN(b)) {
            return b | FP64_QUIET_MASK;
        }
        return c | FP64_QUIET_MASK;
    }
    bool raw_sub = !(*p_mxcsr & FPEMU_MXCSR_DAZ) &&
                   (FP64_IS_SUBNORMAL(a) || FP64_IS_SUBNORMAL(b) || FP64_IS_SUBNORMAL(c));
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP64_IS_SUBNORMAL(a)) {
            a &= FP64_SIGN_MASK;
        }
        if (FP64_IS_SUBNORMAL(b)) {
            b &= FP64_SIGN_MASK;
        }
        if (FP64_IS_SUBNORMAL(c)) {
            c &= FP64_SIGN_MASK;
        }
    }
    bool invalid_mul =
        (FP64_IS_INF(a) && !(b & FP64_ABS_MASK)) || (FP64_IS_INF(b) && !(a & FP64_ABS_MASK));
    if (invalid_mul) {
        return fma_default_nan64(p_mxcsr);
    }

    uint64_t product_sign =
        ((a ^ b) & FP64_SIGN_MASK) ^ (neg_product ? FP64_SIGN_MASK : 0);
    uint64_t c_eff = c ^ (sub_c ? FP64_SIGN_MASK : 0);
    if (FP64_IS_INF(a) || FP64_IS_INF(b)) {
        uint64_t prod_inf = product_sign | FP64_INF;
        if (FP64_IS_INF(c_eff) && ((prod_inf ^ c_eff) == FP64_SIGN_MASK)) {
            return fma_default_nan64(p_mxcsr);
        }
        if (raw_sub) {
            *p_mxcsr |= FPEMU_MXCSR_DE;
        }
        return prod_inf;
    }
    if (FP64_IS_INF(c_eff)) {
        if (raw_sub) {
            *p_mxcsr |= FPEMU_MXCSR_DE;
        }
        return c_eff;
    }
    if (raw_sub) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }

    bool product_zero = !(a & FP64_ABS_MASK) || !(b & FP64_ABS_MASK);
    bool c_zero = !(c_eff & FP64_ABS_MASK);
    if (product_zero && c_zero) {
        uint32_t rm = (*p_mxcsr >> 13) & 3;
        if (product_sign == (c_eff & FP64_SIGN_MASK)) {
            return product_sign;
        }
        return (rm == 1) ? FP64_SIGN_MASK : 0;
    }

    uint64_t mant_a, mant_b, mant_c;
    int32_t exp_a, exp_b, exp_c;
    decode_finite_fp64(a & FP64_ABS_MASK, &mant_a, &exp_a);
    decode_finite_fp64(b & FP64_ABS_MASK, &mant_b, &exp_b);
    decode_finite_fp64(c_eff & FP64_ABS_MASK, &mant_c, &exp_c);

    __uint128_t product = (__uint128_t)mant_a * mant_b;
    int32_t exp_product = exp_a + exp_b;
    int product_top = (product != 0) ? exp_product + 127 - clz_u128(product)
                                     : -1000000;
    int c_top = (mant_c != 0) ? exp_c + (63 - __builtin_clzll(mant_c))
                              : -1000000;
    int32_t low = exp_product < exp_c ? exp_product : exp_c;
    int32_t high = product_top > c_top ? product_top : c_top;

    int32_t base;
    // Exact path: align both terms without exceeding signed __int128 range. Far path:
    // cancellation is impossible, so bits below the retained window only affect sticky.
    if (high - low <= 120) {
        if (!product) {
            base = exp_c;
        } else if (!mant_c) {
            base = exp_product;
        } else {
            base = low;
        }
    } else {
        base = high - 120;
    }

    __int128_t acc = 0;
    __uint128_t product_term = fma_window_term_u128(product, exp_product, base);
    __uint128_t c_term = fma_window_term_u128(mant_c, exp_c, base);
    fma_signed_add_i128(&acc, product_term, product_sign != 0);
    fma_signed_add_i128(&acc, c_term, (c_eff & FP64_SIGN_MASK) != 0);
    return round_i128_to_fp64(acc, base, p_mxcsr);
}

uint64_t fpemu_fp64_fmadd(uint64_t a, uint64_t b, uint64_t c, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_fma_core(a, b, c, false, false, p_mxcsr);
}

uint64_t fpemu_fp64_fmsub(uint64_t a, uint64_t b, uint64_t c, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_fma_core(a, b, c, false, true, p_mxcsr);
}

uint64_t fpemu_fp64_fnmadd(uint64_t a, uint64_t b, uint64_t c, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_fma_core(a, b, c, true, false, p_mxcsr);
}

uint64_t fpemu_fp64_fnmsub(uint64_t a, uint64_t b, uint64_t c, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_fma_core(a, b, c, true, true, p_mxcsr);
}

uint64_t fpemu_fp64_div(uint64_t a, uint64_t b, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    // NaN short-circuit follows the same leading-operand convention as other binops: a wins,
    // with an sNaN raising #I and qNaN propagating.
    uint64_t nan_result;
    if (quiet_binary_nan64(a, b, p_mxcsr, &nan_result)) {
        return nan_result;
    }

    bool raw_sub = !(*p_mxcsr & FPEMU_MXCSR_DAZ) && (FP64_IS_SUBNORMAL(a) || FP64_IS_SUBNORMAL(b));

    if (FP64_IS_INF(a) && FP64_IS_INF(b)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return FP64_DEFAULT_NAN;
    }

    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP64_IS_SUBNORMAL(a)) {
            a &= FP64_SIGN_MASK;
        }
        if (FP64_IS_SUBNORMAL(b)) {
            b &= FP64_SIGN_MASK;
        }
    }

    if (!(a & FP64_ABS_MASK) && !(b & FP64_ABS_MASK)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return FP64_DEFAULT_NAN;
    }

    uint64_t sign_result = (a ^ b) & FP64_SIGN_MASK;
    if (FP64_IS_INF(a)) {
        if (raw_sub) {
            *p_mxcsr |= FPEMU_MXCSR_DE;
        }
        return sign_result | FP64_INF;
    }

    if (!(b & FP64_ABS_MASK)) {
        *p_mxcsr |= FPEMU_MXCSR_ZE;
        return sign_result | FP64_INF;
    }

    if (raw_sub) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }

    if (!(a & FP64_ABS_MASK)) {
        return sign_result;
    }

    if (FP64_IS_INF(b)) {
        return sign_result;
    }

    int32_t exp_a = (a >> 52) & 2047;
    int32_t exp_b = (b >> 52) & 2047;
    uint64_t mant_a = a & FP64_FRAC_MASK;
    uint64_t mant_b = b & FP64_FRAC_MASK;
    if (exp_a) {
        mant_a |= FP64_IMPLICIT_BIT;
    } else {
        int shift = __builtin_clzll(mant_a) - 11; // 52-bit mantissa -> align to bit 52
        mant_a <<= shift;
        exp_a = 1 - shift;
    }
    if (exp_b) {
        mant_b |= FP64_IMPLICIT_BIT;
    } else {
        int shift = __builtin_clzll(mant_b) - 11;
        mant_b <<= shift;
        exp_b = 1 - shift;
    }

    // Compute one extra bit of quotient precision so the LSB of the scaled quotient lands on
    // the rounder guard bit; the remainder becomes the inexact sticky bit.
    __uint128_t num = (__uint128_t)mant_a << 54;
    uint64_t q = (uint64_t)(num / mant_b);
    uint64_t rem = (uint64_t)(num % mant_b);
    __uint128_t mant_result = ((__uint128_t)q << 51) | (rem ? 1ull : 0ull);
    int32_t exp_result = exp_a - exp_b + 1022;
    return round_to_fp64(mant_result, exp_result, sign_result, p_mxcsr);
}

uint64_t fpemu_fp64_sqrt(uint64_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    if (FP64_IS_NAN(a)) {
        if (!(a & FP64_QUIET_MASK)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        return a | FP64_QUIET_MASK;
    }
    if ((*p_mxcsr & FPEMU_MXCSR_DAZ) && FP64_IS_SUBNORMAL(a)) {
        a &= FP64_SIGN_MASK;
    }
    if (!(a & FP64_ABS_MASK)) {
        return a;
    }
    if (a == FP64_INF) {
        return a;
    }
    if (a & FP64_SIGN_MASK) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return FP64_DEFAULT_NAN;
    }
    if (FP64_IS_SUBNORMAL(a)) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }

    int32_t raw_exp = (a >> 52) & 2047;
    uint64_t mant_bits = a & FP64_FRAC_MASK;
    uint64_t mant;
    int32_t unbiased_exp;
    if (raw_exp) {
        mant = mant_bits | FP64_IMPLICIT_BIT;
        unbiased_exp = raw_exp - 1023;
    } else {
        int shift = __builtin_clzll(mant_bits) - 11;
        mant = mant_bits << shift;
        unbiased_exp = 1 - shift - 1023;
    }

    uint64_t scaled = ((unbiased_exp & 1) == 0) ? mant : (mant << 1);
    __uint128_t rem;
    uint64_t root = isqrt128((__uint128_t)scaled << 52, &rem);

    // Feed round_to_fp64. Shift root up by 1 so its MSB sits at bit 53 (round_to_fp64's expected
    // implicit-bit position). rem is the squared-input residue from the integer sqrt; guard is
    // set when rem exceeds root (the round-to-nearest half-way threshold), and sticky is set for
    // any inexact remainder.
    __uint128_t mant_result = ((__uint128_t)root << 1);
    mant_result <<= 52;
    if (rem != 0) {
        mant_result |= 1ull;
    }
    if (rem > root) {
        mant_result |= ((__uint128_t)1ull << 52);
    }

    int32_t e_r_unbiased =
        (unbiased_exp & 1) ? ((unbiased_exp - 1) / 2) : (unbiased_exp / 2);
    int32_t exp_result = e_r_unbiased + 1022;
    return round_to_fp64(mant_result, exp_result, 0, p_mxcsr);
}

uint32_t fpemu_fp64_to_fp32(uint64_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    uint32_t sign_result = (a >> 32) & FP32_SIGN_MASK;
    if (FP64_IS_NAN(a)) {
        if (!(a & FP64_QUIET_MASK)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        // Quietize by ORing in the quiet bit and mapping the top 22 payload bits.
        return sign_result | FP32_QNAN | ((a >> (52 - 23)) & 0x3FFFFF);
    }
    a &= FP64_ABS_MASK;
    if (a == FP64_INF) {
        return sign_result | FP32_INF;
    }
    // Raise the denormal-operand exception if the input is a true FP64 subnormal and DAZ is off.
    if (FP64_IS_SUBNORMAL(a) && !(*p_mxcsr & FPEMU_MXCSR_DAZ)) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (a < FP64_IMPLICIT_BIT) { // treat subnormal or zero magnitude as zero
            a = 0;
        }
    }

    // Decode the finite FP64 source and let round_to_fp32 handle the destination precision and
    // range. Subnormals use biased exponent 1 with no implicit bit.
    uint32_t exp_a = a >> 52;
    uint64_t mant_a = a & FP64_FRAC_MASK;
    if (exp_a) {
        mant_a |= FP64_IMPLICIT_BIT;
    } else if (!mant_a) {
        return sign_result;
    } else {
        exp_a = 1;
    }
    uint64_t mant_result = mant_a; // 1.52
    int32_t exp_result = exp_a - 1023 + 127;
    return round_to_fp32(mant_result, exp_result, sign_result, p_mxcsr);
}

uint64_t fpemu_fp32_to_fp64(uint32_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    // FP32 -> FP64 is always exact: every FP32 value (including subnormals) is exactly
    // representable as a normal FP64. Therefore #P, #O, #U never fire; FTZ is inert. Only #D
    // (subnormal operand, DAZ off) and #I (sNaN) are reachable.
    uint64_t sign_result = (uint64_t)(a & FP32_SIGN_MASK) << 32;
    if (FP32_IS_NAN(a)) {
        if (!(a & FP32_QUIET_MASK)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        // Payload mapping: FP32 mantissa bits 22..0 go to FP64 bits 51..29 (shift left by 29).
        // Quiet bit (FP64 bit 51) is forced on. For qNaN, FP32 bit 22 = 1 naturally lands at
        // FP64 bit 51, so the OR is idempotent; for sNaN it quietizes.
        return sign_result | FP64_QNAN | ((uint64_t)(a & 0x3FFFFF) << 29);
    }
    if (FP32_IS_INF(a)) {
        return sign_result | FP64_INF;
    }
    if (FP32_IS_SUBNORMAL(a)) {
        if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
            return sign_result; // flushed to signed zero; no #D
        }
        *p_mxcsr |= FPEMU_MXCSR_DE;
        // Normalize: shift until bit 23 is MSB, decrement unbiased exponent by shift. Biased
        // FP64 exp = (1 - shift) - 127 + 1023 = 897 - shift.
        uint32_t mant_bits = a & FP32_FRAC_MASK;
        int shift = __builtin_clz(mant_bits) - 8;
        uint32_t mant_norm = (mant_bits << shift) & FP32_FRAC_MASK;
        uint32_t exp64 = 897 - shift;
        return sign_result | ((uint64_t)exp64 << 52) | ((uint64_t)mant_norm << 29);
    }
    if (!(a & FP32_ABS_MASK)) {
        return sign_result; // signed zero
    }
    // Normal: rebias exponent (+896), shift mantissa left 29.
    uint32_t exp32 = (a >> 23) & 0xFF;
    uint64_t mant32 = a & FP32_FRAC_MASK;
    return sign_result | ((uint64_t)(exp32 + 896) << 52) | (mant32 << 29);
}

uint32_t fpemu_fp16_to_fp32(uint16_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    uint32_t sign = (uint32_t)(a & 0x8000u) << 16;
    uint32_t exp = (a >> 10) & 0x1Fu;
    uint32_t frac = a & 0x3FFu;
    if (exp == 0x1Fu) {
        if (!frac) {
            return sign | FP32_INF;
        }
        if (!(frac & 0x200u)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        return sign | FP32_INF | ((frac | 0x200u) << 13);
    }
    if (exp == 0) {
        if (!frac) {
            return sign;
        }
        int top = 31 - __builtin_clz(frac);
        uint32_t exp32 = (uint32_t)(top + 103);
        uint32_t mant32 = (frac << (23 - top)) & FP32_FRAC_MASK;
        return sign | (exp32 << 23) | mant32;
    }
    return sign | ((exp + 112u) << 23) | (frac << 13);
}

static uint16_t fp32_to_fp16_overflow(uint32_t sign, uint32_t rm, uint32_t *p_mxcsr) {
    *p_mxcsr |= FPEMU_MXCSR_OE | FPEMU_MXCSR_PE;
    if (rm == 0) {
        return (uint16_t)(sign | 0x7C00u);
    } else if (rm == 1) {
        return (uint16_t)(sign ? (sign | 0x7C00u) : (sign | 0x7BFFu));
    } else if (rm == 2) {
        return (uint16_t)(sign ? (sign | 0x7BFFu) : (sign | 0x7C00u));
    }
    return (uint16_t)(sign | 0x7BFFu);
}

uint16_t fpemu_fp32_to_fp16(uint32_t a, uint32_t imm, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    FPEMU_ASSERT(imm <= 7, "VCVTPS2PH imm out of range: imm=0x%x", imm);
    uint32_t sign = (a >> 16) & 0x8000u;
    uint32_t abs_a = a & FP32_ABS_MASK;
    uint32_t rm = round_control_from_imm(imm, *p_mxcsr);
    if (FP32_IS_NAN(a)) {
        if (!(a & FP32_QUIET_MASK)) {
            *p_mxcsr |= FPEMU_MXCSR_IE;
        }
        return (uint16_t)(sign | 0x7C00u | 0x200u | ((a >> 13) & 0x1FFu));
    }
    if (FP32_IS_INF(a)) {
        return (uint16_t)(sign | 0x7C00u);
    }
    if (!abs_a) {
        return (uint16_t)sign;
    }
    if (*p_mxcsr & FPEMU_MXCSR_DAZ) {
        if (FP32_IS_SUBNORMAL(a)) {
            return (uint16_t)sign;
        }
    } else if (FP32_IS_SUBNORMAL(a)) {
        *p_mxcsr |= FPEMU_MXCSR_DE;
    }

    uint32_t exp = (abs_a >> 23) & 0xFFu;
    uint32_t frac = abs_a & FP32_FRAC_MASK;
    int32_t e;
    uint32_t mant;
    if (exp == 0) {
        e = -126;
        mant = frac;
    } else {
        e = (int32_t)exp - 127;
        mant = FP32_IMPLICIT_BIT | frac;
    }

    if (e > 15) {
        return fp32_to_fp16_overflow(sign, rm, p_mxcsr);
    }

    if (e >= -14) {
        uint32_t exp16 = (uint32_t)(e + 15);
        uint32_t sig = mant >> 13;
        uint32_t dropped = mant & 0x1FFFu;
        uint32_t guard = (dropped >> 12) & 1u;
        uint32_t sticky = (dropped & 0xFFFu) != 0;
        bool inexact = dropped != 0;
        uint32_t inc = round_increment(rm, sign != 0, sig & 1u, guard, sticky);
        if (inc) {
            sig++;
            if (sig == 0x800u) {
                sig = 0x400u;
                exp16++;
            }
        }
        if (exp16 >= 31) {
            return fp32_to_fp16_overflow(sign, rm, p_mxcsr);
        }
        if (inexact) {
            *p_mxcsr |= FPEMU_MXCSR_PE;
        }
        return (uint16_t)(sign | (exp16 << 10) | (sig & 0x3FFu));
    }

    int32_t sh = -e - 1;
    uint32_t sig = 0;
    uint32_t guard = 0;
    uint32_t sticky = mant != 0;
    uint32_t round_bits = mant;
    if (sh < 32) {
        sig = mant >> sh;
        round_bits = mant & ((1u << sh) - 1u);
        guard = (mant >> (sh - 1)) & 1u;
        sticky = (round_bits & ((1u << (sh - 1)) - 1u)) != 0;
    }
    bool inexact = (guard | sticky) != 0;
    uint32_t inc = round_increment(rm, sign != 0, sig & 1u, guard, sticky);
    sig += inc;
    if (sig == 0x400u) {
        if (inexact) {
            uint32_t underflow;
            if (rm == 0) {
                underflow = round_bits < (3u << (sh - 2));
            } else {
                underflow = round_bits <= (1u << (sh - 1));
            }
            *p_mxcsr |= underflow ? (FPEMU_MXCSR_UE | FPEMU_MXCSR_PE) : FPEMU_MXCSR_PE;
        }
        return (uint16_t)(sign | 0x400u);
    }
    if (inexact) {
        *p_mxcsr |= FPEMU_MXCSR_UE | FPEMU_MXCSR_PE;
    }
    return (uint16_t)(sign | (sig & 0x3FFu));
}

// Convert a nonnegative 64-bit integer magnitude with a known sign bit into FP32 by feeding
// round_to_fp32 with mant_result pre-aligned so the MSB of `abs_val` lands on bit 53. For
// magnitudes wider than 54 bits we shift down by the excess and OR the lost low bits into a single
// sticky bit at position 0 -- exactly the construction round_to_fp32 already consumes from
// fpemu_fp32_div / fpemu_fp32_sqrt. For magnitudes that fit in 54 bits we shift up instead; the
// clz-based normalize inside round_to_fp32 would do the same work but this way the caller does not
// have to worry about clz>=10. No integer value we feed here can overflow FP32 (max abs 2^63
// < 2^127) nor be tiny (min nonzero = 1 >= min-normal 2^-126), so only #P is reachable.
static uint32_t convert_int_magnitude_to_fp32(uint64_t abs_val, uint32_t sign_result,
                                              uint32_t *p_mxcsr) {
    if (!abs_val) {
        return sign_result; // exact +/-0; sign_result already carries -0 sign if caller set it
    }
    int p = 63 - __builtin_clzll(abs_val); // MSB position
    uint64_t mant_result;
    if (p <= 53) {
        mant_result = abs_val << (53 - p);
    } else {
        int sh = p - 53;
        uint64_t lost = abs_val & ((1ull << sh) - 1);
        mant_result = (abs_val >> sh) | (lost ? 1ull : 0ull);
    }
    // After pre-alignment MSB is at bit 53, so round_to_fp32's shift0 is 0 and the +1 in its
    // exponent adjust turns this into the final biased exponent p + 127.
    int32_t exp_result = 126 + p;
    return round_to_fp32(mant_result, exp_result, sign_result, p_mxcsr);
}

// Full 63-bit integer magnitudes require rounding at FP64 precision (53 mantissa bits) but cannot
// overflow or underflow FP64, so this is a minimal non-tiny, non-overflow rounder. This is the
// FP64-precision analogue of the phase-1 rounding in round_to_fp32.
static uint64_t convert_int_magnitude_to_fp64(uint64_t abs_val, uint64_t sign_result,
                                              uint32_t *p_mxcsr) {
    if (!abs_val) {
        return sign_result;
    }
    int p = 63 - __builtin_clzll(abs_val); // MSB position, 0..63
    uint64_t mant; // 53-bit significand (bit 52 = implicit 1 after normalization)
    bool inexact = false;
    if (p <= 52) {
        mant = abs_val << (52 - p);
    } else {
        int sh = p - 52;
        uint64_t guard = (abs_val >> (sh - 1)) & 1;
        uint64_t sticky = (sh >= 2) ? (abs_val & ((1ull << (sh - 1)) - 1)) : 0;
        mant = abs_val >> sh;
        inexact = guard | (sticky != 0);
        uint32_t rm = (*p_mxcsr >> 13) & 3;
        uint32_t sign_bit = (uint32_t)(sign_result >> 63);
        uint32_t inc = round_increment(rm, sign_bit, (uint32_t)(mant & 1), (uint32_t)guard,
                                       (uint32_t)(sticky ? 1u : 0u));
        if (inc) {
            mant++;
            if (mant == (1ull << 53)) {
                mant >>= 1;
                p++; // carry promotes exponent; cannot overflow FP64 for INT64 inputs
            }
        }
    }
    if (inexact) {
        *p_mxcsr |= FPEMU_MXCSR_PE;
    }
    uint64_t biased = (uint64_t)(p + 1023);
    return sign_result | (biased << 52) | (mant & FP64_FRAC_MASK);
}

static uint32_t fpemu_fp32_to_i32_core(uint32_t a, bool trunc, uint32_t *p_mxcsr) {
    if (FP32_IS_NAN(a) || FP32_IS_INF(a)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return 0x80000000u;
    }
    if ((*p_mxcsr & FPEMU_MXCSR_DAZ) && FP32_IS_SUBNORMAL(a)) {
        a &= FP32_SIGN_MASK;
    }
    if (!(a & FP32_ABS_MASK)) {
        return 0;
    }

    bool sign = (a & FP32_SIGN_MASK) != 0;
    uint32_t exp = (a >> 23) & 255;
    uint32_t frac = a & FP32_FRAC_MASK;
    uint64_t mantissa = exp ? (FP32_IMPLICIT_BIT | (uint64_t)frac) : (uint64_t)frac;
    int32_t shift = exp ? ((int32_t)exp - 150) : -149;
    bool inexact = false;
    bool overflow = false;
    uint64_t abs_val = round_fp_to_uint(mantissa, shift, sign, trunc, p_mxcsr, &inexact, &overflow);
    if (overflow || (!sign && abs_val > 0x7FFFFFFFull) || (sign && abs_val > 0x80000000ull)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return 0x80000000u;
    }
    if (inexact) {
        *p_mxcsr |= FPEMU_MXCSR_PE;
    }
    uint64_t mag = sign ? (0ull - abs_val) : abs_val;
    return (uint32_t)mag;
}

uint32_t fpemu_fp32_to_i32(uint32_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_to_i32_core(a, false, p_mxcsr);
}

uint32_t fpemu_fp32_to_i32_trunc(uint32_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_to_i32_core(a, true, p_mxcsr);
}

static uint64_t fpemu_fp32_to_i64_core(uint32_t a, bool trunc, uint32_t *p_mxcsr) {
    if (FP32_IS_NAN(a) || FP32_IS_INF(a)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return FP64_SIGN_MASK;
    }
    if ((*p_mxcsr & FPEMU_MXCSR_DAZ) && FP32_IS_SUBNORMAL(a)) {
        a &= FP32_SIGN_MASK;
    }
    if (!(a & FP32_ABS_MASK)) {
        return 0;
    }

    bool sign = (a & FP32_SIGN_MASK) != 0;
    uint32_t exp = (a >> 23) & 255;
    uint32_t frac = a & FP32_FRAC_MASK;
    uint64_t mantissa = exp ? (FP32_IMPLICIT_BIT | (uint64_t)frac) : (uint64_t)frac;
    int32_t shift = exp ? ((int32_t)exp - 150) : -149;
    bool inexact = false;
    bool overflow = false;
    uint64_t abs_val = round_fp_to_uint(mantissa, shift, sign, trunc, p_mxcsr, &inexact, &overflow);
    if (overflow || (!sign && abs_val > FP64_ABS_MASK) || (sign && abs_val > FP64_SIGN_MASK)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return FP64_SIGN_MASK;
    }
    if (inexact) {
        *p_mxcsr |= FPEMU_MXCSR_PE;
    }
    return sign ? (0ull - abs_val) : abs_val;
}

uint64_t fpemu_fp32_to_i64(uint32_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_to_i64_core(a, false, p_mxcsr);
}

uint64_t fpemu_fp32_to_i64_trunc(uint32_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp32_to_i64_core(a, true, p_mxcsr);
}

static uint32_t fpemu_fp64_to_i32_core(uint64_t a, bool trunc, uint32_t *p_mxcsr) {
    if (FP64_IS_NAN(a) || FP64_IS_INF(a)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return 0x80000000u;
    }
    if ((*p_mxcsr & FPEMU_MXCSR_DAZ) && FP64_IS_SUBNORMAL(a)) {
        a &= FP64_SIGN_MASK;
    }
    if (!(a & FP64_ABS_MASK)) {
        return 0;
    }

    bool sign = (a & FP64_SIGN_MASK) != 0;
    uint32_t exp = (a >> 52) & 2047u;
    uint64_t frac = a & FP64_FRAC_MASK;
    uint64_t mantissa = exp ? (FP64_IMPLICIT_BIT | frac) : frac;
    int32_t shift = exp ? ((int32_t)exp - 1075) : -1074;
    bool inexact = false;
    bool overflow = false;
    uint64_t abs_val = round_fp_to_uint(mantissa, shift, sign, trunc, p_mxcsr, &inexact, &overflow);
    if (overflow || (!sign && abs_val > 0x7FFFFFFFull) || (sign && abs_val > 0x80000000ull)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return 0x80000000u;
    }
    if (inexact) {
        *p_mxcsr |= FPEMU_MXCSR_PE;
    }
    uint64_t mag = sign ? (0ull - abs_val) : abs_val;
    return (uint32_t)mag;
}

uint32_t fpemu_fp64_to_i32(uint64_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_to_i32_core(a, false, p_mxcsr);
}

uint32_t fpemu_fp64_to_i32_trunc(uint64_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_to_i32_core(a, true, p_mxcsr);
}

static uint64_t fpemu_fp64_to_i64_core(uint64_t a, bool trunc, uint32_t *p_mxcsr) {
    if (FP64_IS_NAN(a) || FP64_IS_INF(a)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return FP64_SIGN_MASK;
    }
    if ((*p_mxcsr & FPEMU_MXCSR_DAZ) && FP64_IS_SUBNORMAL(a)) {
        a &= FP64_SIGN_MASK;
    }
    if (!(a & FP64_ABS_MASK)) {
        return 0;
    }

    bool sign = (a & FP64_SIGN_MASK) != 0;
    uint32_t exp = (a >> 52) & 2047u;
    uint64_t frac = a & FP64_FRAC_MASK;
    uint64_t mantissa = exp ? (FP64_IMPLICIT_BIT | frac) : frac;
    int32_t shift = exp ? ((int32_t)exp - 1075) : -1074;
    bool inexact = false;
    bool overflow = false;
    uint64_t abs_val = round_fp_to_uint(mantissa, shift, sign, trunc, p_mxcsr, &inexact, &overflow);
    if (overflow || (!sign && abs_val > FP64_ABS_MASK) || (sign && abs_val > FP64_SIGN_MASK)) {
        *p_mxcsr |= FPEMU_MXCSR_IE;
        return FP64_SIGN_MASK;
    }
    if (inexact) {
        *p_mxcsr |= FPEMU_MXCSR_PE;
    }
    return sign ? (0ull - abs_val) : abs_val;
}

uint64_t fpemu_fp64_to_i64(uint64_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_to_i64_core(a, false, p_mxcsr);
}

uint64_t fpemu_fp64_to_i64_trunc(uint64_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    return fpemu_fp64_to_i64_core(a, true, p_mxcsr);
}

uint32_t fpemu_int32_to_fp32(int32_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    uint32_t sign_result = (a < 0) ? FP32_SIGN_MASK : 0u;
    uint64_t abs_val = abs_i32_to_u64(a);
    return convert_int_magnitude_to_fp32(abs_val, sign_result, p_mxcsr);
}

uint32_t fpemu_int64_to_fp32(int64_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    uint32_t sign_result = (a < 0) ? FP32_SIGN_MASK : 0u;
    uint64_t abs_val = abs_i64_to_u64(a);
    return convert_int_magnitude_to_fp32(abs_val, sign_result, p_mxcsr);
}

uint64_t fpemu_int32_to_fp64(int32_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    // INT32 is always exact in FP64 (32 significant bits fit in 53-bit mantissa). No MXCSR flags
    // are ever raised.
    (void)p_mxcsr;
    if (!a) {
        return 0;
    }
    uint64_t sign_result = (a < 0) ? FP64_SIGN_MASK : 0ull;
    uint64_t abs_val = abs_i32_to_u64(a);
    int p = 63 - __builtin_clzll(abs_val);
    uint64_t mant = abs_val << (52 - p);
    uint64_t biased = (uint64_t)(p + 1023);
    return sign_result | (biased << 52) | (mant & FP64_FRAC_MASK);
}

uint64_t fpemu_int64_to_fp64(int64_t a, uint32_t *p_mxcsr) {
    fpemu_check_mxcsr(p_mxcsr);
    uint64_t sign_result = (a < 0) ? FP64_SIGN_MASK : 0ull;
    uint64_t abs_val = abs_i64_to_u64(a);
    return convert_int_magnitude_to_fp64(abs_val, sign_result, p_mxcsr);
}
