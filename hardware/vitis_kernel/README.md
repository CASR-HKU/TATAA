# Vitis Kernel Setup

In TATAA implementation, we set two RTL kernels for final FPGA design. In other words, we separate memory IO kernel and computation processing kernel in differnet RTL kernels, for better timing optimization. You should generate in each folder `tata_int8os_mem` and `tata_int8os_proc`.

The packaged `.xo` files will be used in the host.

## Steps:

**The steps work for both RTL kernels.**

1. Create a Vivado project, and run the `add_files.tcl`, `add_ips.tcl` in the tcl console.

2. Package RTL kernel in current project, which will lead to a new window for intermediate project.

3. Run the `config_kernel.tcl` in the new project to configure the kernel with AXI interfaces.

4. Package the kernel as a `.xo` file.

## Reference:

[Vitis RTL Kernel Design](https://xilinx.github.io/Vitis-Tutorials/2020-1/docs/build/html/docs/Hardware_Accelerators/Feature_Tutorials/01-rtl_kernel_workflow/README.html#:~:text=This%20tutorial%20demonstrates%20how%20to%20package%20RTL%20IPs,package%20that%20IP%20as%20a%20Vitis%20kernel%20%28.xo%29)

[Packaging the RTL Code as a Vitis XO](https://docs.amd.com/r/en-US/ug1701-vitis-accelerated-embedded/Creating-User-Managed-RTL-Kernels)