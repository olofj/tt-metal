// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/common/math.hpp"
using namespace tt::tt_metal;

// Starting L1 address of commands
inline uint32_t get_command_start_l1_address(bool use_eth_l1) {
    return (use_eth_l1 ? eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE : L1_UNRESERVED_BASE);
}

// Where issue queue interface core pulls in data (follows command)
inline uint32_t get_data_section_l1_address(bool use_eth_l1) {
    uint32_t l1_base = use_eth_l1 ? eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE : L1_UNRESERVED_BASE;
    return l1_base + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
}

// Space available in issue queue interface core pushing command data to consumer to dispatch or relay forward
inline uint32_t get_producer_data_buffer_size(bool use_eth_l1) {
    uint32_t l1_size = use_eth_l1 ? MEM_ETH_SIZE : MEM_L1_SIZE;
    uint32_t l1_base = use_eth_l1 ? eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE : L1_UNRESERVED_BASE;
    return (l1_size - (DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t)) - l1_base);
}

// Space available in core receiving command data to dispatch or relay forward
inline uint32_t get_consumer_data_buffer_size(bool use_eth_l1) {
    uint32_t num_consumer_cmd_slots = use_eth_l1 ? 1 : 2;
    uint32_t producer_data_buffer_size = get_producer_data_buffer_size(use_eth_l1);
    return (producer_data_buffer_size - ((num_consumer_cmd_slots - 1) * DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND)) / num_consumer_cmd_slots;
}

/// @brief Get offset of the command queue relative to its channel
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t relative offset
inline uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size) {
    return cq_id * cq_size;
}

/// @brief Get absolute offset of the command queue
/// @param channel uint16_t channel ID (hugepage)
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t absolute offset
inline uint32_t get_absolute_cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size) {
    return (DeviceCommand::MAX_HUGEPAGE_SIZE * channel) + get_relative_cq_offset(cq_id, cq_size);
}

template <bool addr_16B>
inline uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_ISSUE_READ_PTR + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

template <bool addr_16B>
inline uint32_t get_cq_completion_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_COMPLETION_WRITE_PTR + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

struct SystemMemoryCQInterface {
    // CQ is split into issue and completion regions
    // Host writes commands and data for H2D transfers in the issue region, device reads from the issue region
    // Device signals completion and writes data for D2H transfers in the completion region, host reads from the completion region
    // Equation for issue fifo size is
    // | issue_fifo_wr_ptr + command size B - issue_fifo_rd_ptr |
    // Space available would just be issue_fifo_limit - issue_fifo_size
    SystemMemoryCQInterface(uint16_t channel, uint8_t cq_id, uint32_t cq_size):
      command_issue_region_size(tt::round_up((cq_size - CQ_START) * this->default_issue_queue_split, 32)),
      command_completion_region_size((cq_size - CQ_START) - this->command_issue_region_size),
      issue_fifo_size(command_issue_region_size >> 4),
      issue_fifo_limit(((CQ_START + this->command_issue_region_size) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4),
      completion_fifo_size(command_completion_region_size >> 4),
      completion_fifo_limit(issue_fifo_limit + completion_fifo_size),
      offset(get_absolute_cq_offset(channel, cq_id, cq_size))
     {
        TT_ASSERT(this->issue_fifo_limit != 0, "Cannot have a 0 fifo limit");
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        this->issue_fifo_wr_ptr = (CQ_START + this->offset) >> 4;  // In 16B words
        this->issue_fifo_wr_toggle = 0;

        this->completion_fifo_rd_ptr = this->issue_fifo_limit;
        this->completion_fifo_rd_toggle = 0;
    }

    // Percentage of the command queue that is dedicated for issuing commands. Issue queue size is rounded to be 32B aligned and remaining space is dedicated for completion queue
    // Smaller issue queues can lead to more stalls for applications that send more work to device than readback data.
    static constexpr float default_issue_queue_split = 0.75;
    const uint32_t command_issue_region_size;
    const uint32_t command_completion_region_size;

    uint32_t issue_fifo_size;
    uint32_t issue_fifo_limit;  // Last possible FIFO address
    const uint32_t offset;
    uint32_t issue_fifo_wr_ptr;
    bool issue_fifo_wr_toggle;

    uint32_t completion_fifo_size;
    uint32_t completion_fifo_limit;  // Last possible FIFO address
    uint32_t completion_fifo_rd_ptr;
    bool completion_fifo_rd_toggle;
};

class SystemMemoryManager {
   private:
    chip_id_t device_id;
    const uint32_t m_dma_buf_size;
    const std::function<void(uint32_t, uint32_t, const uint8_t*, uint32_t)> fast_write_callable;
    vector<uint32_t> issue_byte_addrs;
    vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start;
    vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size;
    uint32_t channel_offset;

   public:
    SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs) :
        device_id(device_id),
        m_dma_buf_size(tt::Cluster::instance().get_m_dma_buf_size(device_id)),
        fast_write_callable(
            tt::Cluster::instance().get_fast_pcie_static_tlb_write_callable(device_id)) {

        this->issue_byte_addrs.resize(num_hw_cqs);
        this->completion_byte_addrs.resize(num_hw_cqs);

        // Split hugepage into however many pieces as there are CQs
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        char* hugepage_start = (char*) tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
        this->cq_sysmem_start = hugepage_start;
        this->cq_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / num_hw_cqs;
        this->channel_offset = DeviceCommand::MAX_HUGEPAGE_SIZE * channel;

        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            tt_cxy_pair issue_queue_reader_core = dispatch_core_manager::get(num_hw_cqs).issue_queue_reader_core(device_id, channel, cq_id);
            const std::tuple<uint32_t, uint32_t> issue_interface_tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(issue_queue_reader_core.chip, tt::get_physical_core_coordinate(issue_queue_reader_core, CoreType::WORKER))).value();
            auto [issue_tlb_offset, issue_tlb_size] = issue_interface_tlb_data;
            this->issue_byte_addrs[cq_id] = issue_tlb_offset + CQ_ISSUE_WRITE_PTR % issue_tlb_size;

            tt_cxy_pair completion_queue_writer_core = dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(device_id, channel, cq_id);
            const std::tuple<uint32_t, uint32_t> completion_interface_tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(completion_queue_writer_core.chip, tt::get_physical_core_coordinate(completion_queue_writer_core, CoreType::WORKER))).value();
            auto [completion_tlb_offset, completion_tlb_size] = completion_interface_tlb_data;
            this->completion_byte_addrs[cq_id] = completion_tlb_offset + CQ_COMPLETION_READ_PTR % completion_tlb_size;

            this->cq_interfaces.push_back(SystemMemoryCQInterface(channel, cq_id, this->cq_size));
        }
    }

    void reset(const uint8_t cq_id) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = 0;
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = 0;
    }

    void set_issue_queue_size(const uint8_t cq_id, const uint32_t issue_queue_size) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_size = (issue_queue_size >> 4);
        cq_interface.issue_fifo_limit = (CQ_START + cq_interface.offset + issue_queue_size) >> 4;
    }

    uint32_t get_issue_queue_size(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].issue_fifo_size << 4;
    }

    uint32_t get_issue_queue_limit(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].issue_fifo_limit << 4;
    }

    uint32_t get_completion_queue_size(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_size << 4;
    }

    uint32_t get_completion_queue_limit(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_limit << 4;
    }

    uint32_t get_issue_queue_write_ptr(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].issue_fifo_wr_ptr << 4;
    }

    uint32_t get_completion_queue_read_ptr(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_rd_ptr << 4;
    }

    void issue_queue_reserve_back(uint32_t cmd_size_B, const uint8_t cq_id) const {
        uint32_t cmd_size_16B = align(cmd_size_B, 32) >> 4;

        uint32_t rd_ptr_and_toggle;
        uint32_t rd_ptr;
        uint32_t rd_toggle;

        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        do {
            rd_ptr_and_toggle = get_cq_issue_rd_ptr<true>(this->device_id, cq_id, this->cq_size);
            rd_ptr = rd_ptr_and_toggle & 0x7fffffff;
            rd_toggle = rd_ptr_and_toggle >> 31;
        } while (
            cq_interface
                .issue_fifo_wr_ptr < rd_ptr and cq_interface.issue_fifo_wr_ptr + cmd_size_16B > rd_ptr or

            // This is the special case where we wrapped our wr ptr and our rd ptr
            // has not yet moved
            (rd_toggle != cq_interface.issue_fifo_wr_toggle and cq_interface.issue_fifo_wr_ptr == rd_ptr));
    }

    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) const {
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        //  this->cq_sysmem_start gives start of hugepage for a given channel
        //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
        //  so channel offset needs to be subtracted to get address relative to channel
        // TODO: Reconsider offset sysmem offset calculations based on https://github.com/tenstorrent-metal/tt-metal/issues/4757
        void* user_scratchspace = this->cq_sysmem_start + (write_ptr - this->channel_offset);

        memcpy(user_scratchspace, data, size_in_bytes);
    }

    void send_issue_queue_write_ptr(const uint8_t cq_id) const {
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        uint32_t write_ptr_and_toggle =
            cq_interface.issue_fifo_wr_ptr | (cq_interface.issue_fifo_wr_toggle << 31);
        this->fast_write_callable(this->issue_byte_addrs[cq_id], 4, (uint8_t*)&write_ptr_and_toggle, this->m_dma_buf_size);
        tt_driver_atomics::sfence();
    }

    void issue_queue_push_back(uint32_t push_size_B, bool lazy, const uint8_t cq_id) {
        // All data needs to be 32B aligned

        uint32_t push_size_16B = align(push_size_B, 32) >> 4;

        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        cq_interface.issue_fifo_wr_ptr += push_size_16B;

        if (cq_interface.issue_fifo_wr_ptr >= cq_interface.issue_fifo_limit) {
            cq_interface.issue_fifo_wr_ptr -= cq_interface.issue_fifo_size;

            // Flip the toggle
            cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
        }

        // Notify dispatch core
        if (not lazy) {
            this->send_issue_queue_write_ptr(cq_id);
        }
    }

    void completion_queue_wait_front(const uint8_t cq_id) {
        uint32_t write_ptr_and_toggle;
        uint32_t write_ptr;
        uint32_t write_toggle;
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        do {
            write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(this->device_id, cq_id, this->cq_size);
            write_ptr = write_ptr_and_toggle & 0x7fffffff;
            write_toggle = write_ptr_and_toggle >> 31;
        } while (cq_interface.completion_fifo_rd_ptr == write_ptr and cq_interface.completion_fifo_rd_toggle == write_toggle);
    }

    void send_completion_queue_read_ptr(const uint8_t cq_id) const {
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        uint32_t read_ptr_and_toggle =
            cq_interface.completion_fifo_rd_ptr | (cq_interface.completion_fifo_rd_toggle << 31);
        this->fast_write_callable(this->completion_byte_addrs[cq_id], 4, (uint8_t*)&read_ptr_and_toggle, this->m_dma_buf_size);
        tt_driver_atomics::sfence();
    }

    void wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
        this->send_issue_queue_write_ptr(cq_id);
    }

    void wrap_completion_queue_rd_ptr(const uint8_t cq_id) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    void completion_queue_pop_front(uint32_t data_read_B, const uint8_t cq_id) {
        uint32_t data_read_16B = align(data_read_B, 32) >> 4;

        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr += data_read_16B;
        if (cq_interface.completion_fifo_rd_ptr >= cq_interface.completion_fifo_limit) {
            cq_interface.completion_fifo_rd_ptr = cq_interface.command_issue_region_size >> 4;
            cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
        }

        // Notify dispatch core
        this->send_completion_queue_read_ptr(cq_id);
    }

};
