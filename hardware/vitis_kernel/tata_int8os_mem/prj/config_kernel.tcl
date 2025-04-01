ipx::associate_bus_interfaces -busif m_axi_instr -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axi_yizo -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axi_xi -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]
set ipx_reg [ipx::add_register CTRL [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property address_offset 0x00 $ipx_reg
set_property size 32 $ipx_reg
set ipx_reg [ipx::add_register AXI_INSTR_BASE_ADDR [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property address_offset 0x10 $ipx_reg
set_property size 64 $ipx_reg
set ipx_reg [ipx::add_register INSTR_BTT [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property address_offset 0x18 $ipx_reg
set_property size 32 $ipx_reg
set ipx_reg [ipx::add_register AXI_YIZO_BASE_ADDR [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property address_offset 0x1c $ipx_reg
set_property size 64 $ipx_reg
set ipx_reg [ipx::add_register AXI_XI_BASE_ADDR [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property address_offset 0x24 $ipx_reg
set_property size 64 $ipx_reg
set ipx_reg [ipx::add_register CORE_DEBUG_STATUS [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property address_offset 0x2c $ipx_reg
set_property size 32 $ipx_reg
set ipx_reg [ipx::add_register CORE_LATENCY_CYCLES [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property address_offset 0x30 $ipx_reg
set_property size 32 $ipx_reg
set ipx_reg [ipx::add_register CORE_INSTR_STATUS [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property address_offset 0x34 $ipx_reg
set_property size 32 $ipx_reg
set ipx_reg [ipx::add_register CORE_MEM_ITF_STATUS [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property address_offset 0x38 $ipx_reg
set_property size 32 $ipx_reg
ipx::associate_bus_interfaces -busif m_axis_mm2s_xi -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axis_mm2s_yi -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axis_pccmd -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axis_pcfbk -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axis_s2mm_zo -clock ap_clk [ipx::current_core]
ipx::add_register_parameter ASSOCIATED_BUSIF [ipx::get_registers AXI_INSTR_BASE_ADDR -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property value          {m_axi_instr}          [ipx::get_register_parameters ASSOCIATED_BUSIF     \
                                    -of_objects [ipx::get_registers AXI_INSTR_BASE_ADDR                      \
                                    -of_objects [ipx::get_address_blocks reg0                      \
                                    -of_objects [ipx::get_memory_maps s_axi_control                 \
                                    -of_objects [ipx::current_core]]]]]
ipx::add_register_parameter ASSOCIATED_BUSIF [ipx::get_registers AXI_YIZO_BASE_ADDR -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property value          {m_axi_yizo}          [ipx::get_register_parameters ASSOCIATED_BUSIF     \
                                    -of_objects [ipx::get_registers AXI_YIZO_BASE_ADDR                      \
                                    -of_objects [ipx::get_address_blocks reg0                      \
                                    -of_objects [ipx::get_memory_maps s_axi_control                 \
                                    -of_objects [ipx::current_core]]]]]
ipx::add_register_parameter ASSOCIATED_BUSIF [ipx::get_registers AXI_XI_BASE_ADDR -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property value          {m_axi_xi}          [ipx::get_register_parameters ASSOCIATED_BUSIF     \
                                    -of_objects [ipx::get_registers AXI_XI_BASE_ADDR                      \
                                    -of_objects [ipx::get_address_blocks reg0                      \
                                    -of_objects [ipx::get_memory_maps s_axi_control                 \
                                    -of_objects [ipx::current_core]]]]]