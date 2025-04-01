create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 -module_name axi_datamover_yizo
set_property -dict [list CONFIG.Component_Name {axi_datamover_yizo} CONFIG.c_m_axi_mm2s_data_width {256} CONFIG.c_m_axis_mm2s_tdata_width {256} CONFIG.c_mm2s_burst_size {128} CONFIG.c_mm2s_btt_used {23} CONFIG.c_m_axi_s2mm_data_width {256} CONFIG.c_s_axis_s2mm_tdata_width {256} CONFIG.c_s2mm_burst_size {128} CONFIG.c_s2mm_btt_used {23} CONFIG.c_mm2s_include_sf {false} CONFIG.c_s2mm_include_sf {false} CONFIG.c_m_axi_mm2s_id_width {0} CONFIG.c_m_axi_s2mm_id_width {0} CONFIG.c_enable_cache_user {true} CONFIG.c_addr_width {40}] [get_ips axi_datamover_yizo]
generate_target {instantiation_template} [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_yizo/axi_datamover_yizo.xci]
set_property generate_synth_checkpoint false [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_yizo/axi_datamover_yizo.xci]
generate_target all [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_yizo/axi_datamover_yizo.xci]
export_ip_user_files -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_yizo/axi_datamover_yizo.xci] -no_script -sync -force -quiet
export_simulation -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_yizo/axi_datamover_yizo.xci] -directory ../prj/tata_int8os_mem.ip_user_files/sim_scripts -ip_user_files_dir ../prj/tata_int8os_mem.ip_user_files -ipstatic_source_dir ../prj/tata_int8os_mem.ip_user_files/ipstatic -lib_map_path [list {modelsim=../prj/tata_int8os_mem.cache/compile_simlib/modelsim} {questa=../prj/tata_int8os_mem.cache/compile_simlib/questa} {ies=../prj/tata_int8os_mem.cache/compile_simlib/ies} {xcelium=../prj/tata_int8os_mem.cache/compile_simlib/xcelium} {vcs=../prj/tata_int8os_mem.cache/compile_simlib/vcs} {riviera=../prj/tata_int8os_mem.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
update_compile_order -fileset sources_1
create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 -module_name axi_datamover_xi
set_property -dict [list CONFIG.Component_Name {axi_datamover_xi} CONFIG.c_m_axi_mm2s_data_width {256} CONFIG.c_m_axis_mm2s_tdata_width {256} CONFIG.c_mm2s_burst_size {128} CONFIG.c_mm2s_btt_used {23} CONFIG.c_include_s2mm {Omit} CONFIG.c_include_s2mm_stsfifo {false} CONFIG.c_s2mm_addr_pipe_depth {3} CONFIG.c_mm2s_include_sf {false} CONFIG.c_s2mm_include_sf {false} CONFIG.c_m_axi_mm2s_id_width {0} CONFIG.c_m_axi_s2mm_awid {1} CONFIG.c_enable_cache_user {true} CONFIG.c_enable_s2mm {0} CONFIG.c_addr_width {40}] [get_ips axi_datamover_xi]
generate_target {instantiation_template} [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_xi/axi_datamover_xi.xci]
set_property generate_synth_checkpoint false [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_xi/axi_datamover_xi.xci]
generate_target all [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_xi/axi_datamover_xi.xci]
export_ip_user_files -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_xi/axi_datamover_xi.xci] -no_script -sync -force -quiet
export_simulation -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_xi/axi_datamover_xi.xci] -directory ../prj/tata_int8os_mem.ip_user_files/sim_scripts -ip_user_files_dir ../prj/tata_int8os_mem.ip_user_files -ipstatic_source_dir ../prj/tata_int8os_mem.ip_user_files/ipstatic -lib_map_path [list {modelsim=../prj/tata_int8os_mem.cache/compile_simlib/modelsim} {questa=../prj/tata_int8os_mem.cache/compile_simlib/questa} {ies=../prj/tata_int8os_mem.cache/compile_simlib/ies} {xcelium=../prj/tata_int8os_mem.cache/compile_simlib/xcelium} {vcs=../prj/tata_int8os_mem.cache/compile_simlib/vcs} {riviera=../prj/tata_int8os_mem.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
update_compile_order -fileset sources_1
create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 -module_name axi_datamover_instr
set_property -dict [list CONFIG.Component_Name {axi_datamover_instr} CONFIG.c_m_axi_mm2s_data_width {64} CONFIG.c_m_axis_mm2s_tdata_width {64} CONFIG.c_mm2s_burst_size {256} CONFIG.c_mm2s_btt_used {23} CONFIG.c_include_s2mm {Omit} CONFIG.c_include_s2mm_stsfifo {false} CONFIG.c_s2mm_addr_pipe_depth {3} CONFIG.c_mm2s_include_sf {false} CONFIG.c_s2mm_include_sf {false} CONFIG.c_m_axi_mm2s_id_width {0} CONFIG.c_m_axi_s2mm_awid {1} CONFIG.c_enable_cache_user {true} CONFIG.c_enable_s2mm {0} CONFIG.c_addr_width {40}] [get_ips axi_datamover_instr]
generate_target {instantiation_template} [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_instr/axi_datamover_instr.xci]
set_property generate_synth_checkpoint false [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_instr/axi_datamover_instr.xci]
generate_target all [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_instr/axi_datamover_instr.xci]
export_ip_user_files -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_instr/axi_datamover_instr.xci] -no_script -sync -force -quiet
export_simulation -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axi_datamover_instr/axi_datamover_instr.xci] -directory ../prj/tata_int8os_mem.ip_user_files/sim_scripts -ip_user_files_dir ../prj/tata_int8os_mem.ip_user_files -ipstatic_source_dir ../prj/tata_int8os_mem.ip_user_files/ipstatic -lib_map_path [list {modelsim=../prj/tata_int8os_mem.cache/compile_simlib/modelsim} {questa=../prj/tata_int8os_mem.cache/compile_simlib/questa} {xcelium=../prj/tata_int8os_mem.cache/compile_simlib/xcelium} {vcs=../prj/tata_int8os_mem.cache/compile_simlib/vcs} {riviera=../prj/tata_int8os_mem.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
update_compile_order -fileset sources_1
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_256
set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.Component_Name {axis_register_256}] [get_ips axis_register_256]
generate_target {instantiation_template} [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_256/axis_register_256.xci]
set_property generate_synth_checkpoint false [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_256/axis_register_256.xci]
generate_target all [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_256/axis_register_256.xci]
export_ip_user_files -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_256/axis_register_256.xci] -no_script -sync -force -quiet
export_simulation -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_256/axis_register_256.xci] -directory ../prj/tata_int8os_mem.ip_user_files/sim_scripts -ip_user_files_dir ../prj/tata_int8os_mem.ip_user_files -ipstatic_source_dir ../prj/tata_int8os_mem.ip_user_files/ipstatic -lib_map_path [list {modelsim=../prj/tata_int8os_mem.cache/compile_simlib/modelsim} {questa=../prj/tata_int8os_mem.cache/compile_simlib/questa} {ies=../prj/tata_int8os_mem.cache/compile_simlib/ies} {xcelium=../prj/tata_int8os_mem.cache/compile_simlib/xcelium} {vcs=../prj/tata_int8os_mem.cache/compile_simlib/vcs} {riviera=../prj/tata_int8os_mem.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
update_compile_order -fileset sources_1
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_32
set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.Component_Name {axis_register_32}] [get_ips axis_register_32]
generate_target {instantiation_template} [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_32/axis_register_32.xci]
set_property generate_synth_checkpoint false [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_32/axis_register_32.xci]
generate_target all [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_32/axis_register_32.xci]
export_ip_user_files -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_32/axis_register_32.xci] -no_script -sync -force -quiet
export_simulation -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_32/axis_register_32.xci] -directory ../prj/tata_int8os_mem.ip_user_files/sim_scripts -ip_user_files_dir ../prj/tata_int8os_mem.ip_user_files -ipstatic_source_dir ../prj/tata_int8os_mem.ip_user_files/ipstatic -lib_map_path [list {modelsim=../prj/tata_int8os_mem.cache/compile_simlib/modelsim} {questa=../prj/tata_int8os_mem.cache/compile_simlib/questa} {ies=../prj/tata_int8os_mem.cache/compile_simlib/ies} {xcelium=../prj/tata_int8os_mem.cache/compile_simlib/xcelium} {vcs=../prj/tata_int8os_mem.cache/compile_simlib/vcs} {riviera=../prj/tata_int8os_mem.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
update_compile_order -fileset sources_1
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_8
set_property -dict [list CONFIG.Component_Name {axis_register_8}] [get_ips axis_register_8]
generate_target {instantiation_template} [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_8/axis_register_8.xci]
set_property generate_synth_checkpoint false [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_8/axis_register_8.xci]
generate_target all [get_files  ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_8/axis_register_8.xci]
export_ip_user_files -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_8/axis_register_8.xci] -no_script -sync -force -quiet
export_simulation -of_objects [get_files ../prj/tata_int8os_mem.srcs/sources_1/ip/axis_register_8/axis_register_8.xci] -directory ../prj/tata_int8os_mem.ip_user_files/sim_scripts -ip_user_files_dir ../prj/tata_int8os_mem.ip_user_files -ipstatic_source_dir ../prj/tata_int8os_mem.ip_user_files/ipstatic -lib_map_path [list {modelsim=../prj/tata_int8os_mem.cache/compile_simlib/modelsim} {questa=../prj/tata_int8os_mem.cache/compile_simlib/questa} {ies=../prj/tata_int8os_mem.cache/compile_simlib/ies} {xcelium=../prj/tata_int8os_mem.cache/compile_simlib/xcelium} {vcs=../prj/tata_int8os_mem.cache/compile_simlib/vcs} {riviera=../prj/tata_int8os_mem.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
update_compile_order -fileset sources_1