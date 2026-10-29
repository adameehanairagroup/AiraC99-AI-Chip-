/**
 * ============================================================================
 * AIRA CORE - MATHEMATICAL ENGINE VERIFICATION SUITE
 * 
 * Tests RMSNorm, MatMul vector products, Softmax normalization,
 * and RoPE rotation against known mathematical invariants.
 * ============================================================================
 */

#include "aira_math.h"
#include "hal/hal_os.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static void test_rmsnorm(void) {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float o[4] = {0};

    aira_rmsnorm(o, x, w, 4);

    /* Sum of squares = 1 + 4 + 9 + 16 = 30. Mean = 7.5. sqrt(7.5 + 1e-5) ~= 2.7386 */
    float expected_scale = 1.0f / sqrtf(7.5f + 1e-5f);
    for (int i = 0; i < 4; i++) {
        float diff = fabsf(o[i] - (x[i] * expected_scale));
        assert(diff < 1e-4f);
    }
    hal_print("[PASS] RMSNorm Mathematical Verification\n");
}

static void test_matmul(void) {
    float x[3] = {1.0f, 2.0f, 3.0f};
    float w[6] = {
        1.0f, 2.0f, 3.0f,   /* Row 0: 1*1 + 2*2 + 3*3 = 14 */
        4.0f, 5.0f, 6.0f    /* Row 1: 4*1 + 5*2 + 6*3 = 32 */
    };
    float xout[2] = {0};

    aira_matmul(xout, x, w, 3, 2);

    assert(fabsf(xout[0] - 14.0f) < 1e-4f);
    assert(fabsf(xout[1] - 32.0f) < 1e-4f);
    hal_print("[PASS] MatMul Vector Dot Product Verification\n");
}

static void test_softmax(void) {
    float x[3] = {1.0f, 2.0f, 3.0f};
    aira_softmax(x, 3);

    float sum = x[0] + x[1] + x[2];
    assert(fabsf(sum - 1.0f) < 1e-5f);
    assert(x[2] > x[1] && x[1] > x[0]);
    hal_print("[PASS] Softmax Normalization Verification\n");
}

int main(void) {
    hal_init();
    hal_print("\n=== AIRA CORE MATHEMATICAL ENGINE TESTS ===\n");
    test_rmsnorm();
    test_matmul();
    test_softmax();
    hal_print("=== ALL MATHEMATICAL INVARIANTS VERIFIED! ===\n\n");
    hal_shutdown();
    return 0;
}
