/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include "dataflow_api.h"

template <bool DRAM>
InterleavedAddrGenFast<DRAM> InterleavedAddrGenFastHelper_(uint32_t addr, tt::CB cb, uint32_t idx) {
    uint32_t tile_bytes = get_tile_size(cb);
    auto data_format = get_dataformat(cb);

    const InterleavedAddrGenFast<DRAM> x = {
        .bank_base_address = addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    return x;
}

#define InterleavedAddrGenFastHelper(addr, cb, idx) \
    ({ \
        constexpr bool is_dram = (get_compile_time_arg_val(idx) == 1); \
        const InterleavedAddrGenFast<is_dram> ret = InterleavedAddrGenFastHelper_<is_dram>(addr, cb, idx); \
        ret; \
    })


template<typename AddrGen>
void noc_async_read_tile_helper(tt::CB cb, uint32_t num_tiles, uint32_t tile_idx, AddrGen addr_gen) {
    cb_reserve_back(cb, num_tiles);
    uint32_t addr = get_write_ptr(cb);
    noc_async_read_tile(tile_idx, addr_gen, addr);
    noc_async_read_barrier();
    cb_push_back(cb, num_tiles);
}

template<typename AddrGen>
void noc_async_write_tile_helper(tt::CB cb, uint32_t num_tiles, uint32_t tile_idx, AddrGen addr_gen) {
    cb_wait_front(cb, num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb);
    noc_async_write_tile(tile_idx, addr_gen, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb, num_tiles);
}

void generate_bcast_scaler(
    uint32_t cb_scaler,
    uint32_t scaler) {
    union { float f; uint32_t u; } u; u.u = scaler;
    cb_reserve_back(cb_scaler, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_scaler));

    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[k*256 + j] = uint16_t(u.u>>16);
    cb_push_back(cb_scaler, 1);
}

void fill_cb_with_value(uint32_t cb_id, uint32_t value) {
    cb_reserve_back(cb_id, 1);
    auto ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_id));
    for (int j = 0; j < 1024; j++) {
        ptr[j] = uint16_t(value >> 16);
    }
    cb_push_back(cb_id, 1);
}

void generate_mask_w(
    uint32_t cb_mask,
    uint32_t mask_w) {
    union { float f; uint32_t u; } one; one.f = 1.0f;
    union { float f; uint32_t u; } zero; zero.f = 0.0f;

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask));

    for(uint32_t h = 0 ; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16) mask_w_0 = 16;
            uint32_t w = 0;
            for(; w < mask_w_0; w++){
                ptr[h * 16 + w] = uint16_t(one.u >> 16);
            }
            for(; w < 16; w++){
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for(; w < mask_w_1; w++){
                ptr[h * 16 + w + 256] = uint16_t(one.u >> 16);
            }
            for(; w < 16; w++){
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 2
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16) mask_w_0 = 16;
            uint32_t w = 0;
            for(; w < mask_w_0; w++){
                ptr[h * 16 + w + 512] = uint16_t(one.u >> 16);
            }
            for(; w < 16; w++){
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for(; w < mask_w_1; w++){
                ptr[h * 16 + w + 768] = uint16_t(one.u >> 16);
            }
            for(; w < 16; w++){
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }

    cb_push_back(cb_mask, 1);
}

void generate_mask_h(
    uint32_t cb_mask,
    uint32_t mask_h) {
    union { float f; uint32_t u; } one; one.f = 1.0f;
    union { float f; uint32_t u; } zero; zero.f = 0.0f;

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask));

    for(uint32_t w = 0; w < 16; w++) {
        // sub tile 0
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16) mask_h_0 = 16;
            uint32_t h = 0;
            for(; h < mask_h_0; h++){
                ptr[h * 16 + w] = uint16_t(one.u >> 16);
            }
            for(; h < 16; h++){
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 1
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16) mask_h_0 = 16;
            uint32_t h = 0;
            for(; h < mask_h_0; h++){
                ptr[h * 16 + w + 256] = uint16_t(one.u >> 16);
            }
            for(; h < 16; h++){
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 2
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for(; h < mask_h_1; h++){
                ptr[h * 16 + w + 512] = uint16_t(one.u >> 16);
            }
            for(; h < 16; h++){
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 3
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for(; h < mask_h_1; h++){
                ptr[h * 16 + w + 768] = uint16_t(one.u >> 16);
            }
            for(; h < 16; h++){
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }

    cb_push_back(cb_mask, 1);
}
