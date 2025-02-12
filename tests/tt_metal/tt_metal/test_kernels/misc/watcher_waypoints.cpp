// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ckernel.h"
#include "debug/status.h"

// Helper function to sync execution by forcing other riscs to wait for brisc, which in turn waits
// for a set number of cycles.
void hacky_sync(uint32_t sync_num, uint32_t wait_cycles, uint32_t sync_addr) {
    volatile tt_l1_ptr uint32_t* sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_addr);
#if defined(COMPILE_FOR_BRISC)
    ckernel::wait(wait_cycles);
    *(sync_ptr) = sync_num;
#else
    while (*(sync_ptr) != sync_num) { ; }
#endif
}

/*
 * A test for the watcher waypointing feature.
*/
#if defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_NCRISC)
void kernel_main() {
#else
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#endif
    uint32_t sync_wait_cycles = get_arg_val<uint32_t>(0);
    uint32_t sync_address     = get_arg_val<uint32_t>(1);

    // Post a new waypoint with a delay after (to let the watcher poll it)
    hacky_sync(1, sync_wait_cycles, sync_address);
    DEBUG_STATUS('A', 'A', 'A', 'A');
    hacky_sync(2, sync_wait_cycles, sync_address);
    DEBUG_STATUS('B', 'B', 'B', 'B');
    hacky_sync(3, sync_wait_cycles, sync_address);
    DEBUG_STATUS('C', 'C', 'C', 'C');
    hacky_sync(4, sync_wait_cycles, sync_address);
#if defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_NCRISC)
}
#else
}
}
#endif
