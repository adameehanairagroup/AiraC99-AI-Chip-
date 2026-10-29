/**
 * ============================================================================
 * AIRA CORE - MATHEMATICAL OPERATIONS IMPLEMENTATION
 * 
 * Highly optimized vector math routines for transformer inference.
 * Features ARM NEON / AVX2 vector acceleration with auto-fallback to portable C.
 * 
 * Architect: Er. Adam Eehan (Aira Group of Technology)
 * Co-Pilot: Aira 3 Ultra (Sovereign Engine v3.6)
 * ============================================================================
 */

#include "aira_math.h"
#include <math.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define USE_NEON 1
#endif

void aira_rmsnorm(float* o, const float* x, const float* weight, int size) {
    float ss = 0.0f;

    #if defined(USE_NEON)
    int i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i <= size - 4; i += 4) {
        float32x4_t xv = vld1q_f32(&x[i]);
        sum_vec = vmlaq_f32(sum_vec, xv, xv);
    }
    ss = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
         vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
    for (; i < size; i++) {
        ss += x[i] * x[i];
    }
    #else
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    #endif

    ss /= (float)size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    #if defined(USE_NEON)
    float32x4_t scale_vec = vdupq_n_f32(ss);
    i = 0;
    for (; i <= size - 4; i += 4) {
        float32x4_t xv = vld1q_f32(&x[i]);
        float32x4_t wv = vld1q_f32(&weight[i]);
        float32x4_t norm_x = vmulq_f32(xv, scale_vec);
        float32x4_t out_v = vmulq_f32(wv, norm_x);
        vst1q_f32(&o[i], out_v);
    }
    for (; i < size; i++) {
        o[i] = weight[i] * (x[i] * ss);
    }
    #else
    for (int i = 0; i < size; i++) {
        o[i] = weight[i] * (x[i] * ss);
    }
    #endif
}

void aira_matmul(float* xout, const float* x, const float* w, int n, int d) {
    /* W is (d, n), x is (n), xout is (d) */
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const float* w_row = w + (i * n);

        #if defined(USE_NEON)
        int j = 0;
        float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
        float32x4_t sum_vec1 = vdupq_n_f32(0.0f);

        for (; j <= n - 8; j += 8) {
            float32x4_t xv0 = vld1q_f32(&x[j]);
            float32x4_t wv0 = vld1q_f32(&w_row[j]);
            sum_vec0 = vmlaq_f32(sum_vec0, xv0, wv0);

            float32x4_t xv1 = vld1q_f32(&x[j + 4]);
            float32x4_t wv1 = vld1q_f32(&w_row[j + 4]);
            sum_vec1 = vmlaq_f32(sum_vec1, xv1, wv1);
        }
        float32x4_t total_vec = vaddq_f32(sum_vec0, sum_vec1);
        val = vgetq_lane_f32(total_vec, 0) + vgetq_lane_f32(total_vec, 1) +
              vgetq_lane_f32(total_vec, 2) + vgetq_lane_f32(total_vec, 3);

        for (; j < n; j++) {
            val += w_row[j] * x[j];
        }
        #else
        for (int j = 0; j < n; j++) {
            val += w_row[j] * x[j];
        }
        #endif

        xout[i] = val;
    }
}

void aira_matmul_q8_0(float* xout, const float* x, const block_q8_0* w, int n, int d) {
    int blocks_per_row = n / 32;
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const block_q8_0* row = w + (i * blocks_per_row);

        for (int b = 0; b < blocks_per_row; b++) {
            float sum_b = 0.0f;
            const float* x_block = x + (b * 32);
            for (int k = 0; k < 32; k++) {
                sum_b += x_block[k] * (float)row[b].qs[k];
            }
            val += sum_b * row[b].scale;
        }
        xout[i] = val;
    }
}

void aira_rope(float* q, float* k, int pos, int dim, int head_size, int n_heads, int n_kv_heads) {
    (void)dim;
    /* Rotary Position Embedding applied to Q and K heads */
    for (int h = 0; h < n_heads; h++) {
        float* q_head = q + (h * head_size);
        for (int i = 0; i < head_size; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
            float val = (float)pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            float q0 = q_head[i];
            float q1 = q_head[i + 1];
            q_head[i]     = q0 * fcr - q1 * fci;
            q_head[i + 1] = q0 * fci + q1 * fcr;
        }
    }

    for (int h = 0; h < n_kv_heads; h++) {
        float* k_head = k + (h * head_size);
        for (int i = 0; i < head_size; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
            float val = (float)pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            float k0 = k_head[i];
            float k1 = k_head[i + 1];
            k_head[i]     = k0 * fcr - k1 * fci;
            k_head[i + 1] = k0 * fci + k1 * fcr;
        }
    }
}

void aira_softmax(float* x, int size) {
    if (size <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

void aira_swiglu(float* hb, const float* hb2, int size) {
    for (int i = 0; i < size; i++) {
        float val = hb[i];
        /* SiLU(x) = x * sigmoid(x) */
        val = val * (1.0f / (1.0f + expf(-val)));
        hb[i] = val * hb2[i];
    }
}

void aira_accumulate(float* a, const float* b, int size) {
    #if defined(USE_NEON)
    int i = 0;
    for (; i <= size - 4; i += 4) {
        float32x4_t av = vld1q_f32(&a[i]);
        float32x4_t bv = vld1q_f32(&b[i]);
        vst1q_f32(&a[i], vaddq_f32(av, bv));
    }
    for (; i < size; i++) {
        a[i] += b[i];
    }
    #else
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
    #endif
}
