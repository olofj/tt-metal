// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_untilize_api.h"
#endif

namespace ckernel {

/**
 * Init function for untilize operations, to be used at the beginning of the kernel.
 */
ALWI void untilize_init(uint32_t icb, uint32_t ocb = 16)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(0, 0, icb) ));
    MATH(( llk_math_pack_sync_init<SyncHalf>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(ocb) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_untilize_hw_configure_disaggregated(icb) ));
    UNPACK(( llk_unpack_untilize_init(icb) )); // init must be after configure
}

/**
 * Short init function to initialize untilize op, after a full init is already performed.
 */
ALWI void untilize_init_short(uint32_t icb)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(0, 0, icb) ));
    UNPACK(( llk_unpack_untilize_init(icb) ));
}

/**
 * Perform the untilize operation on a block of tiles. This simply loops over the provided block size.
*/
template <int N = 1>
ALWI void untilize_block(uint32_t icb, uint32_t block, uint32_t ocb)
{
    UNPACK(( llk_unpack_untilize(icb, block) ));

    for (uint32_t t = 0; t < block / N; t++) {
        MATH(( llk_math_wait_for_dest_available<SYNC>() ));

        // Datacopy
        for (int reg_id = 0; reg_id < N; reg_id++) {
            MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(reg_id) ));
        }

        MATH(( llk_math_dest_section_done<SYNC>() ));

        PACK(( llk_packer_wait_for_math_done() ));

        // Datacopy
        for (int reg_id = 0; reg_id < N; reg_id++) {
            PACK(( llk_pack<false, SYNC, false >(reg_id, ocb)  ));
        }

        // Release dest
        PACK(( llk_pack_dest_section_done<SYNC>() ));
    }
}

/**
 * Uninitialize untilize operation, to allow initializing another operation.
 */
ALWI void untilize_uninit(uint32_t icb) {
    UNPACK(( llk_unpack_untilize_uninit(icb) ));
}

}
