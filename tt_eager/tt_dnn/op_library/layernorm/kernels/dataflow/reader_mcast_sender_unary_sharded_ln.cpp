// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// split REDUCE across cores
void kernel_main() {

    constexpr uint32_t reduce_receiver_semaphore_addr = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_addr   = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks                     = get_compile_time_arg_val(2);
    constexpr uint32_t block_h                        = get_compile_time_arg_val(3);
    constexpr uint32_t block_h_size_bytes             = get_compile_time_arg_val(4);
    constexpr uint32_t num_all_to_all_workers          = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_worker            = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_worker_bytes      = get_compile_time_arg_val(7);
    constexpr uint32_t num_tiles_per_worker_last            = get_compile_time_arg_val(8);
    constexpr uint32_t num_tiles_per_worker_last_bytes      = get_compile_time_arg_val(9);
    constexpr bool row_major                                = (bool) get_compile_time_arg_val(10);
    constexpr uint32_t num_x                                = get_compile_time_arg_val(11);
    constexpr uint32_t num_y                                = get_compile_time_arg_val(12);

    const uint32_t mcast_dest_noc_start_x               = get_arg_val<uint32_t>(0);
    const uint32_t mcast_dest_noc_start_y               = get_arg_val<uint32_t>(1);
    const uint32_t mcast_dest_noc_end_x                 = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_end_y                 = get_arg_val<uint32_t>(3);
    const uint32_t start_x                              = get_arg_val<uint32_t>(4);
    const uint32_t start_y                              = get_arg_val<uint32_t>(5);

    volatile tt_l1_ptr uint32_t * in0_remote_noc_x          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(6));
    volatile tt_l1_ptr uint32_t * in0_remote_noc_y          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(6 + num_x));

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0;
    constexpr uint32_t cb_ex = tt::CB::dataflow1;
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3;
    constexpr uint32_t cb_ex2 = tt::CB::dataflow4;
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3;
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);
    const DataFormat data_format = get_dataformat(cb_ex_partial);

    uint64_t remote_noc_addrs[num_blocks];

    uint32_t x = start_x, y = start_y;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        remote_noc_addrs[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
        if constexpr(row_major) {
            ++y;
            if (y == num_y) {
                y = 0;
                ++x;
                if (x == num_x) {
                    x = 0;
                }
            }
        } else {
            ++x;
            if (x == num_x) {
                x = 0;
                ++y;
                if (y == num_y) {
                    y = 0;
                }
            }
        }
    }

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        0);

    const uint64_t reduce_sender_semaphore_noc_addr = multicast_data_noc | reduce_sender_semaphore_addr;

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);

    const auto& global_reduce_sender = [&](const uint32_t cb_partial, const uint32_t cb_external, const uint32_t cb_ex, const uint32_t cb_ex_global) __attribute__((always_inline))
    {
        // global reduce
        // wait for local data ready
        cb_wait_front(cb_partial, block_h);
        // inc semaphore of other cores
        if constexpr(num_blocks > 1) {
            *reduce_sender_semaphore_addr_ptr = VALID;
            noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_blocks-1);
            noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);
            noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks-1);
        }

        // read data from other cores
        uint32_t l1_read_addr_ex_par = get_read_ptr(cb_partial);
        for (uint32_t i = 0; i < num_tiles_per_worker; ++i) {
            uint32_t l1_write_addr_external = get_write_ptr(cb_external);
            for(uint32_t block = 0; block < num_blocks; ++block) {
                cb_reserve_back(cb_external, 1);
                uint64_t noc_addr_ex_par = remote_noc_addrs[block] | l1_read_addr_ex_par;
                noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
                l1_write_addr_external += single_tile_size_bytes;
                noc_async_read_barrier();

                cb_push_back(cb_external, 1);
            }
            l1_read_addr_ex_par += single_tile_size_bytes;
        }

        uint32_t l1_read_addr_ex = get_read_ptr(cb_ex);
        uint32_t l1_write_addr_ex_global = get_write_ptr(cb_ex_global);
        cb_wait_front(cb_ex, num_tiles_per_worker);

        // sync with other workers
        if constexpr(num_blocks > 1) {
            noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_blocks-1);
            noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);
            noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks-1);
        }

        // gather data to top row
        cb_reserve_back(cb_ex_global, block_h);
        for (uint32_t block = 0; block < num_all_to_all_workers; ++block) {
            uint64_t noc_addr_ex = remote_noc_addrs[block] | l1_read_addr_ex;
            uint32_t num_tiles = block == num_all_to_all_workers - 1 ? num_tiles_per_worker_last_bytes : num_tiles_per_worker_bytes;
            if constexpr (num_tiles_per_worker_bytes <= NOC_MAX_BURST_SIZE)
                noc_async_read_one_packet(noc_addr_ex, l1_write_addr_ex_global, num_tiles);
            else
                noc_async_read(noc_addr_ex, l1_write_addr_ex_global, num_tiles);
            l1_write_addr_ex_global += num_tiles;
        }
        noc_async_read_barrier();

        // mcast
        uint32_t l1_read_addr_ex_global = get_read_ptr(cb_ex_global);

        if constexpr(num_blocks > 1) {
            for (uint32_t block = 0; block < num_all_to_all_workers; ++block) {
                *reduce_sender_semaphore_addr_ptr = block + 1;

                uint32_t num_tiles = block == num_all_to_all_workers - 1 ? num_tiles_per_worker_last_bytes : num_tiles_per_worker_bytes;

                noc_async_write_multicast(l1_read_addr_ex_global, multicast_data_noc | l1_read_addr_ex_global, num_tiles, num_blocks-1, true);

                if (block == num_all_to_all_workers-1)
                    noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks-1, false);
                else
                    noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks-1, true);

                l1_read_addr_ex_global += num_tiles;
                noc_async_write_barrier();
            }
        }
        cb_push_back(cb_ex_global, block_h);
    };

    global_reduce_sender(cb_ex_partial, cb_ex_external, cb_ex, cb_ex_global);
    global_reduce_sender(cb_ex_partial2, cb_ex_external2, cb_ex2pe, cb_ex_global);
}
