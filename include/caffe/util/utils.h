#ifndef JPEG2PNG_UTILS_H
#define JPEG2PNG_UTILS_H

#include <assert.h>
#include <stdnoreturn.h>
#include <time.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

// Convenience macros
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))
#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define DUMP(v, f) do { printf( #v " = " f "\n", v); } while(false)
#define DUMPD(v) DUMP(v, "%d")
#define DUMPF(v) DUMP(v, "%f")
#define DUMP_SIMD(r) do { __m128 _t = r; _mm_empty(); printf( #r " = %.9e,%.9e,%.9e,%.9e\n", _t[0], _t[1], _t[2], _t[3]); } while(false)
#define SWAP(type, x, y) do { type _t = x; x = y; y = _t; } while(false)
#define SQR(x) ((x) * (x))


// asserts
#if defined(NDEBUG) && defined(BUILTIN_UNREACHABLE)
  #define ASSUME(x) do { if(!(x)) { __builtin_unreachable(); } } while(false)
#else
  #define ASSUME(x) assert(x)
#endif
#if defined(NDEBUG) && defined(BUILTIN_ASSUME_ALIGNED)
  #define ASSUME_ALIGNED(x) x = __builtin_assume_aligned(x, 16)
#else
  #define ASSUME_ALIGNED(x) ASSUME((((uintptr_t)x) & 15) == 0)
#endif

// hide some warnings for some unused functions
#ifdef ATTRIBUTE_UNUSED
  #define POSSIBLY_UNUSED __attribute__((unused))
#else
  #define POSSIBLY_UNUSED
#endif

// bounds check
static inline void check(unsigned x, unsigned y, unsigned w, unsigned h) {
        ASSUME(x < w);
        ASSUME(y < h);
        (void) x;
        (void) y;
        (void) w;
        (void) h;
}

// index image with bounds check
static inline float *p(float *in, unsigned x, unsigned y, unsigned w, unsigned h) {
        check(x, y, w, h);
        return &in[(size_t)y * w + x];
}

// convenience
static inline float sqf(float x) {
        return x * x;
}

#endif
