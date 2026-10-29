/**
 * ============================================================================
 * AIRA CORE - MATHEMATICAL OPERATIONS & TENSOR ROUTINES
 * 
 * Optimized vector math for transformer architectures:
 *   - RMSNorm (Root Mean Square Normalization)
 *   - MatMul (General Matrix-Vector Multiplication)
 *   - RoPE (Rotary Position Embeddings)
 *   - SwiGLU / SiLU (Swish Gated Linear Unit)
 *   - Numerically stable Softmax
 * 
 * Architect: Er. Adam Eehan (Aira Group of Technology)
 * ============================================================================
 */

#ifndef AIRA_MATH_H
#define AIRA_MATH_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Root Mean Square Normalization: o = x / sqrt(mean(x^2) + eps) * weight */
void aira_rmsnorm(float* o, const float* x, const float* weight, int size);

/* Matrix-Vector Multiplication: xout = W * x (W is d x n, x is n, xout is d) */
void aira_matmul(float* xout, const float* x, const float* w, int n, int d);

/* Quantized Q8_0 Matrix-Vector Multiplication */
typedef struct {
    float scale;
    int8_t qs[32];
} block_q8_0;

void aira_matmul_q8_0(float* xout, const float* x, const block_q8_0* w, int n, int d);

/* Rotary Position Embedding (RoPE) */
void aira_rope(float* q, float* k, int pos, int dim, int head_size, int n_heads, int n_kv_heads);

/* Softmax in-place with numerical stability (subtract max) */
void aira_softmax(float* x, int size);

/* SwiGLU: hb[i] = hb[i] * (1.0f / (1.0f + expf(-hb[i]))) * hb2[i] */
void aira_swiglu(float* hb, const float* hb2, int size);

/* Vector Accumulation: a[i] += b[i] */
void aira_accumulate(float* a, const float* b, int size);

#ifdef __cplusplus
}
#endif

#endif /* AIRA_MATH_H */
