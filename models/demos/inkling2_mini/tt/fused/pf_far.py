# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill far-cache write at a row read on the device (data movement only).

fill_cache(kv_full, rows, update_idx=s) bakes s into the program's runtime args, so a captured trace writes the
same rows on every replay. This kernel reads s from DRAM (the pass's `spos` buffer), so one trace serves every
pass start the host writes there.

Call: PfFar(device)(rows, cache, spos) with rows [1, H, T, W] and cache [1, H, S, W], same TILE dtype in interleaved
DRAM, T and s multiples of 32; spos: uint32 [1, 16] ROW_MAJOR whose first value is s (s + T <= S).
"""

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

NB = 8  # tiles per batch
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


class PfFar:
    def __init__(self, device):
        self.device = device

    @staticmethod
    def ok(rows, cache):
        dram = lambda x: x.memory_config().buffer_type == ttnn.BufferType.DRAM and not x.is_sharded()  # noqa: E731
        return (
            rows.dtype == cache.dtype in TILE_BYTES
            and rows.layout == cache.layout == ttnn.TILE_LAYOUT
            and dram(rows)
            and dram(cache)
            and rows.shape[0] == cache.shape[0] == 1
            and rows.shape[1] == cache.shape[1]
            and rows.shape[3] == cache.shape[3]
            and rows.shape[2] % 32 == 0
            and rows.shape[2] <= cache.shape[2]
        )

    def __call__(self, rows, cache, spos):
        assert self.ok(rows, cache), (rows, cache)
        assert spos.dtype == ttnn.uint32 and spos.layout == ttnn.ROW_MAJOR_LAYOUT and spos.shape[-1] == 16, spos
        h, t, w = rows.shape[1], rows.shape[2], rows.shape[3]
        total = h * (t // 32) * (w // 32)
        grid = self.device.compute_with_storage_grid_size()
        n_cores = max(1, min(grid.x * grid.y, total // NB))
        cores, crs = F.core_list(self.device, n_cores)
        rt = {
            core: [spos.buffer_address(), rows.buffer_address(), cache.buffer_address(),
                   total * i // n_cores, total * (i + 1) // n_cores]
            for i, core in enumerate(cores)
        }  # fmt: skip
        tb = TILE_BYTES[rows.dtype]
        named = dict(t32=t // 32, c32=cache.shape[2] // 32, wt=w // 32, nb=NB, tb=tb)
        k = F.kernel("pf_far.cpp", ttnn.ReaderConfigDescriptor(), named, rt, crs)
        cbs = [F.cb(0, rows.dtype, NB, tb, cores=crs), F.cb(1, ttnn.bfloat16, 1, 64, cores=crs)]
        F.run([spos, rows, cache], [k], cbs)
