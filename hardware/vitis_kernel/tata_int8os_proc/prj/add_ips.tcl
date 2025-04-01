create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_256
set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.Component_Name {axis_register_256}] [get_ips axis_register_256]
generate_target {instantiation_template} [get_files /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_256/axis_register_256.xci]
set_property generate_synth_checkpoint false [get_files  /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_256/axis_register_256.xci]
generate_target all [get_files  /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_256/axis_register_256.xci]
export_ip_user_files -of_objects [get_files /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_256/axis_register_256.xci] -no_script -sync -force -quiet
export_simulation -of_objects [get_files /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_256/axis_register_256.xci] -directory /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts -ip_user_files_dir /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files -ipstatic_source_dir /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/ipstatic -lib_map_path [list {modelsim=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/modelsim} {questa=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/questa} {ies=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/ies} {xcelium=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/xcelium} {vcs=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/vcs} {riviera=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
update_compile_order -fileset sources_1
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_32
set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.Component_Name {axis_register_32}] [get_ips axis_register_32]
generate_target {instantiation_template} [get_files /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_32/axis_register_32.xci]
set_property generate_synth_checkpoint false [get_files  /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_32/axis_register_32.xci]
generate_target all [get_files  /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_32/axis_register_32.xci]
export_ip_user_files -of_objects [get_files /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_32/axis_register_32.xci] -no_script -sync -force -quiet
export_simulation -of_objects [get_files /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_32/axis_register_32.xci] -directory /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts -ip_user_files_dir /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files -ipstatic_source_dir /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/ipstatic -lib_map_path [list {modelsim=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/modelsim} {questa=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/questa} {ies=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/ies} {xcelium=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/xcelium} {vcs=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/vcs} {riviera=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
update_compile_order -fileset sources_1
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_8
set_property -dict [list CONFIG.Component_Name {axis_register_8}] [get_ips axis_register_8]
generate_target {instantiation_template} [get_files /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_8/axis_register_8.xci]
set_property generate_synth_checkpoint false [get_files  /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_8/axis_register_8.xci]
generate_target all [get_files  /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_8/axis_register_8.xci]
export_ip_user_files -of_objects [get_files /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_8/axis_register_8.xci] -no_script -sync -force -quiet
export_simulation -of_objects [get_files /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.srcs/sources_1/ip/axis_register_8/axis_register_8.xci] -directory /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/sim_scripts -ip_user_files_dir /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files -ipstatic_source_dir /vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.ip_user_files/ipstatic -lib_map_path [list {modelsim=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/modelsim} {questa=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/questa} {ies=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/ies} {xcelium=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/xcelium} {vcs=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/vcs} {riviera=/vol/datastore/tata_vitis/tata_int8os_proc/prj/tata_int8os_proc.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
update_compile_order -fileset sources_1