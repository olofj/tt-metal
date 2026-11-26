// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill far-cache write (data movement only): rows [1, H, T, W] TILE into cache [1, H, S, W] at row s, with s
// read from DRAM (the spos buffer's first uint32), so one program and one captured trace serve every s.
// Source tile p = (h * T32 + r) * WT + c goes to cache tile (h * C32 + s / 32 + r) * WT + c.
// Runtime args: spos addr, rows addr, cache addr, first tile, end tile.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T32 = get_named_compile_time_arg_val("t32");  // rows tile rows (T / 32)
    constexpr uint32_t C32 = get_named_compile_time_arg_val("c32");  // cache tile rows (S / 32)
    constexpr uint32_t WT = get_named_compile_time_arg_val("wt");    // tile columns
    constexpr uint32_t NB = get_named_compile_time_arg_val("nb");    // tiles per batch
    constexpr uint32_t TB = get_named_compile_time_arg_val("tb");    // tile bytes
    constexpr uint32_t cb_a = 0, cb_s = 1;
    const uint32_t la = get_write_ptr(cb_a), ls = get_write_ptr(cb_s);
    const InterleavedAddrGen<true> sp = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 64};
    noc_async_read(sp.get_noc_addr(0), ls, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t s32 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ls) >> 5;
    const InterleavedAddrGen<true> ra = {.bank_base_address = get_arg_val<uint32_t>(1), .page_size = TB};
    const InterleavedAddrGen<true> ca = {.bank_base_address = get_arg_val<uint32_t>(2), .page_size = TB};
    const uint32_t p1 = get_arg_val<uint32_t>(4);
    for (uint32_t p0 = get_arg_val<uint32_t>(3); p0 < p1; p0 += NB) {
        const uint32_t cnt = p1 - p0 < NB ? p1 - p0 : NB;
        for (uint32_t i = 0; i < cnt; ++i) {
            noc_async_read(ra.get_noc_addr(p0 + i), la + i * TB, TB);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < cnt; ++i) {
            const uint32_t p = p0 + i, h = p / (T32 * WT), rem = p - h * (T32 * WT);
            noc_async_write(la + i * TB, ca.get_noc_addr(h * C32 * WT + s32 * WT + rem), TB);
        }
        noc_async_write_barrier();
    }
}
