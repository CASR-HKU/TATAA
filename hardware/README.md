# Hardware Architecture

This directory contains the hardware design source code for the FPGA-based TATAA system implemented on Alveo U280.

## Host

- The path `./host` contains the source code for the host software that runs on the host CPU. The host software is responsible for loading the FPGA bitstream, initializing the FPGA, and communicating with the FPGA to perform the TATAA system functions.

- Note that we only provide an example host design for Transformer inference in this repository. The host software for other applications can be developed based on this example.

- **Before running host codes, you should compile the model to get instruction binary files. See `../compilation/README.md` for more details**

# RTL

All the rtl source codes are in the `./rtl` directory. The RTL design is implemented in Verilog HDL. For hardware implementation details, please refer to our paper published in TRETS.

# Vitis Kernel

For your reference, we provide the vitis kernel projects in the `./vitis_kernel` directory. The vitis kernel generates the RTL kernel for xilinx runtime. Since we use a lot of Xilinx IPs, you should strictly follow Vitis 2023.2 set up to generate the kernel's feature. The steps of how to generate the kernel can be found inside the `./vitis_kernel/README.md`.