#include "libs.h"

int tata_node_op(std::vector<xrt::ip> &ips, uint32_t num_cores, uint32_t instr_len, 
                 uint64_t *instr_addr_arr, uint64_t *x_base_addr, uint64_t *yizo_base_addr) {
    
    uint32_t ap_check = 0xffffffff;
    uint32_t ap_idle[num_cores];
    uint32_t axi_ctrl_rd[num_cores];
    uint32_t debug_status[num_cores];
    uint32_t instr_status[num_cores];
    uint32_t mem_itf_status[num_cores];

    for (uint32_t i = 0; i < num_cores; i++) {

        std::cout << "  Setting Instruction Num" << std::endl;
        ips[i].write_register(ADDR_INSTR_BTT, instr_len * 8);

        std::cout << "  Setting Instruction Address)" << std::endl;
        ips[i].write_register(ADDR_INSTR_BASE_ADDR_0, instr_addr_arr[i]);
        ips[i].write_register(ADDR_INSTR_BASE_ADDR_1, instr_addr_arr[i] >> 32);

        std::cout << "  Setting YIZO Address)" << std::endl;
        ips[i].write_register(ADDR_YIZO_BASE_ADDR_0, yizo_base_addr[i]);
        ips[i].write_register(ADDR_YIZO_BASE_ADDR_1, yizo_base_addr[i] >> 32);

        std::cout << "  Setting XI Address)" << std::endl;
        ips[i].write_register(ADDR_XI_BASE_ADDR_0, x_base_addr[i]);
        ips[i].write_register(ADDR_XI_BASE_ADDR_1, x_base_addr[i] >> 32);

        std::cout << "  Start the IP Core" << std::endl;
        ips[i].write_register(ADDR_AP_CTRL, IP_START);
    }
    std::cout << "  Wait until the IP is DONE" << std::endl;

    while (ap_check != IP_IDLE)
    {
        ap_check = 0xffffffff;
        for (u_int32_t i = 0; i < num_cores; i++)
        {
            axi_ctrl_rd[i] = ips[i].read_register(ADDR_AP_CTRL);
            ap_idle[i] = axi_ctrl_rd[i] & 0xfffffff4;
            ap_check = ap_check & ap_idle[i];
        }
    }

    debug_status[0] = ips[0].read_register(ADDR_CORE_DEBUG_STATUS);
    std::cout << "    Debug Status from IP "<< " 0x" << std::setfill('0') << std::setw(8) << std::hex << debug_status[0] << std::endl;
    mem_itf_status[0] = ips[0].read_register(ADDR_CORE_MEM_ITF_STATUS);
    std::cout << "    MEM_ITF_STATUS: " << " 0x" << std::setfill('0') << std::setw(8) << std::hex << mem_itf_status[0] << std::endl;
    instr_status[0] = ips[0].read_register(ADDR_CORE_INSTR_STATUS);
    std::cout << "    INSTR STATUE: " << " 0x" << std::setfill('0') << std::setw(8) << std::hex << instr_status[0] << std::endl;


    return 1;
}

int tata_node_status(std::vector<xrt::ip> &ips, 
                    uint32_t num_cores, uint32_t freq,
                    uint32_t bfp_mult_num, uint32_t tapu_num, 
                    uint32_t xw, uint32_t xh, uint32_t yw, 
                    uint32_t mac_ops, uint32_t layer_num) {
    uint32_t debug_status[num_cores];
    uint32_t latency_cycles[num_cores];
    uint32_t instr_status[num_cores];
    uint32_t mem_itf_status[num_cores];
    uint32_t ap_ctrl_itf_status[num_cores];
    for (uint32_t i = 0; i < num_cores; i++) {
        // Read the status from IP
        std::cout << "Read Instr Status from IP" << std::endl;
        instr_status[i] = ips[i].read_register(ADDR_CORE_INSTR_STATUS);
        std::cout << "INSTR STATUE: " << i << " 0x" << std::setfill('0') << std::setw(8) << std::hex << instr_status[i] << std::endl;

        std::cout << "Read Status from IP " << i << std::endl;
        debug_status[i] = ips[i].read_register(ADDR_CORE_DEBUG_STATUS);
        std::cout << "DEBUG_STATUS: " << i << " 0x" << std::setfill('0') << std::setw(8) << std::hex << debug_status[i] << std::endl;

        std::cout << "Read Memory Interface Status from IP" << std::endl;
        mem_itf_status[i] = ips[i].read_register(ADDR_CORE_MEM_ITF_STATUS);
        std::cout << "MEM_ITF_STATUS: " << i << " 0x" << std::setfill('0') << std::setw(8) << std::hex << mem_itf_status[i] << std::endl;

        std::cout << "Read AAP Control Status from IP" << std::endl;
        ap_ctrl_itf_status[i] = ips[i].read_register(ADDR_AP_CTRL); 
        std::cout << "ADDR_AP_CTRL_STATUS: " << i << " 0x" << std::setfill('0') << std::setw(8) << std::hex << ap_ctrl_itf_status[i] << std::endl; // 0x00000004 IDLE

        std::cout << "Read Latency Cycles from IP " << i << std::endl;
        latency_cycles[i] = ips[i].read_register(ADDR_CORE_LATENCY_CYCLES);
        std::cout << "LATENCY_CYCLES: " << i << " 0x" << std::setfill('0') << std::setw(8) << std::hex << latency_cycles[i] << std::endl;
    }

    uint32_t max_latency_cycles = 0;
    for (uint32_t i = 0; i < num_cores; i++)
    {
        if (latency_cycles[i] > max_latency_cycles)
        {
            max_latency_cycles = latency_cycles[i];
        }
    }

    float latency_ms = (float)max_latency_cycles / freq * 1000;
    std::cout << "Latency Cycles: "<< std::dec << max_latency_cycles << std::endl;
    std::cout << "Latency (ms): " << latency_ms << std::endl;

    // uint32_t ops =  bfp_mult_num * tapu_num * xw * xh * yw * mac_ops * layer_num * 2;
    // float throughput = (float)ops * num_cores / latency_ms / 1000000;
    // std::cout << "Throughput (GOP/s): " << throughput << std::endl;

    return 1;
}

int instr_load(char **fp_list, uint64_t **bo_instr_map, uint32_t num_cores, uint32_t instr_num){

    uint64_t *bf_instr[num_cores];
    for (uint32_t i = 0; i < num_cores; i++)
    {
        bf_instr[i] = new uint64_t[instr_num];
        std::ifstream file(fp_list[i], std::ios::binary | std::ios::ate);
        if (file.is_open())
        {
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            if (file.read(reinterpret_cast<char *>(bf_instr[i]), size)){}
            file.close();
        }
        for (uint32_t r = 0; r < instr_num; r++)
            bo_instr_map[i][r] = bf_instr[i][r];
    }

    return 0;
}

int bo_sync(uint32_t num_cores, 
            xrt::bo *bo_x, xrt::bo *bo_yizo, xrt::bo *bo_instr,
            uint64_t *bo_x_addr, uint64_t *bo_yizo_addr, uint64_t *bo_instr_addr) {
    for (uint32_t i = 0; i < num_cores; i++)
    {
        bo_instr_addr[i] = bo_instr[i].address();
        bo_x_addr[i] = bo_x[i].address();
        bo_yizo_addr[i] = bo_yizo[i].address();

        bo_x[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_yizo[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_instr[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    return 0;
}

int map_bo_gen(uint32_t num_cores, uint32_t hbm_ch_num, 
                uint32_t *bank_assign, auto device, 
                uint32_t x_size, uint32_t y_size, uint32_t instr_size,
                xrt::bo *bo_x, xrt::bo *bo_yizo, xrt::bo *bo_instr,
                uint8_t **bo_x_map, uint8_t **bo_yizo_map, uint64_t **bo_instr_map) {

    for (uint32_t i = 0; i < num_cores; i++)
    {
        bo_x[i] = xrt::bo(device, x_size, bank_assign[i * hbm_ch_num]);
        bo_yizo[i] = xrt::bo(device, y_size, bank_assign[i * hbm_ch_num + 1]);
        bo_instr[i] = xrt::bo(device, instr_size, DDR0); // DDR[0] (xbutil examine --report memory)
        
        bo_x_map[i] = bo_x[i].map<uint8_t *>();
        bo_yizo_map[i] = bo_yizo[i].map<uint8_t *>();
        bo_instr_map[i] = bo_instr[i].map<uint64_t *>();

        std::fill(bo_x_map[i], bo_x_map[i] + x_size, 127);
        std::fill(bo_yizo_map[i], bo_yizo_map[i] + y_size, 63);
    }

    return 0;
}

