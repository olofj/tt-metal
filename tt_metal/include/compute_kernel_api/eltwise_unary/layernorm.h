#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_sqrt.h"
#include "llk_math_eltwise_unary_sfpu_recip.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

ALWI void sqrt_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sqrt_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void sqrt_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sqrt<APPROX, SyncHalf>(idst) ));
}

ALWI void recip_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_reciprocal_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void recip_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_reciprocal<APPROX, SyncHalf>(idst) ));
}

} // namespace ckernel
