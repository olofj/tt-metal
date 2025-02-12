
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_fixture.hpp"
#include "test_utils.hpp"
#include "llrt/llrt.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher waypoints.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

static void RunTest(WatcherFixture* fixture, Device* device) {
    // Set up program
    Program program = Program();

    // Set up dram buffers
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 50;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    uint32_t l1_buffer_addr = 400 * 1024;


    tt_metal::InterleavedBufferConfig dram_config{
                            .device=device,
                            .size = dram_buffer_size,
                            .page_size = dram_buffer_size,
                            .buffer_type = tt_metal::BufferType::DRAM
                            };
    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer.address();

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer.address();

    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    // A DRAM copy kernel, we'll feed it incorrect inputs to test sanitization.
    constexpr CoreCoord core = {0, 0}; // Run kernel on first core
    auto dram_copy_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    // Write to the input DRAM buffer
    std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::detail::WriteToBuffer(input_dram_buffer, input_vec);

    // Write runtime args - update to a core that doesn't exist
    output_dram_noc_xy.x = 16;
    output_dram_noc_xy.y = 16;
    tt_metal::SetRuntimeArgs(
        program,
        dram_copy_kernel,
        core,
        {l1_buffer_addr,
        input_dram_buffer_addr,
        (std::uint32_t)input_dram_noc_xy.x,
        (std::uint32_t)input_dram_noc_xy.y,
        output_dram_buffer_addr,
        (std::uint32_t)output_dram_noc_xy.x,
        (std::uint32_t)output_dram_noc_xy.y,
        dram_buffer_size});

    // Run the kernel, expect an exception here
    try {
        fixture->RunProgram(device, program);
    } catch (std::runtime_error& e) {
        const string expected = "Command Queue could not finish: device hang due to illegal NoC transaction. See build/watcher.log for details.";
        const string error = string(e.what());
        log_info(tt::LogTest, "Caught exception (one is expected in this test)");
        EXPECT_TRUE(error.find(expected) != string::npos);
    }

    // We should be able to find the expected watcher error in the log as well.
    CoreCoord phys_core = device->worker_core_from_logical_core(core);
    string expected = "Device x, Core (x=x,y=x):    NAWW,*,*,*,*  brisc using noc0 tried to access core (16,16) L1[addr=0x00019020,len=102400]";
    expected[7] = '0' + device->id();
    expected[18] = '0' + phys_core.x;
    expected[22] = '0' + phys_core.y;
    EXPECT_TRUE(
        FileContainsAllStrings(
            fixture->log_file_name,
            {expected}
        )
    );
}

// Run tests for host-side sanitization (uses functions that are from watcher.hpp).
void CheckHostSanitization(Device *device) {
    // Try reading from a core that doesn't exist
    constexpr CoreCoord core = {16, 16};
    uint64_t addr = 0;
    uint32_t sz_bytes = 4;
    try {
        llrt::read_hex_vec_from_core(device->id(), core, addr, sz_bytes);
    } catch (std::runtime_error& e) {
        const string expected = "Host watcher: bad {} NOC coord {}\nread\n" + core.str();
        const string error = string(e.what());
        log_info(tt::LogTest, "Caught exception (one is expected in this test)");
        EXPECT_TRUE(error.find(expected) != string::npos);
    }
}

TEST_F(WatcherFixture, TestWatcherSanitize) {
    // Skip this test for slow dipatch for now. Due to how llrt currently sits below device, it's
    // tricky to check watcher server status from the finish loop for slow dispatch. Once issue #4363
    // is resolved, we should add a check for print server handing in slow dispatch as well.
    if (this->slow_dispatch_)
        GTEST_SKIP();

    CheckHostSanitization(this->devices_[0]);
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(RunTest, device);
    }
}
