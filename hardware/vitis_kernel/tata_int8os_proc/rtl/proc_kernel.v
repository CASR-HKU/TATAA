// TODO: add license
`timescale 1 ns / 1 ps

// Top level of the kernel. Do not modify module name, parameters or ports.
module proc_kernel #(
    parameter integer C_S_AXIS_XYZ_DATA_WIDTH   = 256,
    parameter integer C_S_AXIS_PCCMD_DATA_WIDTH = 32,
    parameter integer C_S_AXIS_PCFBK_DATA_WIDTH = 8
) (
    // System Signals
    input  wire                                 ap_clk,
    input  wire                                 ap_resetn,
    //  Note: A minimum subset of AXI4 memory mapped signals are declared.  AXI
    // signals omitted from these interfaces are automatically inferred with the
    // optimal values for Xilinx accleration platforms.  This allows Xilinx AXI4 Interconnects
    // within the system to be optimized by removing logic for AXI4 protocol
    // features that are not necessary. When adapting AXI4 masters within the RTL
    // kernel that have signals not declared below, it is suitable to add the
    // signals to the declarations below to connect them to the AXI4 Master.
    // 
    // List of ommited signals - effect
    // -------------------------------
    // ID - Transaction ID are used for multithreading and out of order
    // transactions.  This increases complexity. This saves logic and increases Fmax
    // in the system when ommited.
    // SIZE - Default value is log2(data width in bytes). Needed for subsize bursts.
    // This saves logic and increases Fmax in the system when ommited.
    // BURST - Default value (0b01) is incremental.  Wrap and fixed bursts are not
    // recommended. This saves logic and increases Fmax in the system when ommited.
    // LOCK - Not supported in AXI4
    // CACHE - Default value (0b0011) allows modifiable transactions. No benefit to
    // changing this.
    // PROT - Has no effect in current acceleration platforms.
    // QOS - Has no effect in current acceleration platforms.
    // REGION - Has no effect in current acceleration platforms.
    // USER - Has no effect in current acceleration platforms.
    // RESP - Not useful in most acceleration platforms.
    // 
    // AXI stream (master) for micro instructions from memory kernel
    output wire                                 s_axis_pccmd_tready,
    input  wire                                 s_axis_pccmd_tvalid,
    input  wire [C_S_AXIS_PCCMD_DATA_WIDTH-1:0] s_axis_pccmd_tdata,
    // AXI stream (slave) for feedback to memory kernel
    input  wire                                 m_axis_pcfbk_tready,
    output wire                                 m_axis_pcfbk_tvalid,
    output wire [C_S_AXIS_PCFBK_DATA_WIDTH-1:0] m_axis_pcfbk_tdata,
    // AXI stream (master) for data xyz to processing kernel
    output wire                                 s_axis_mm2s_xi_tready,
    input  wire                                 s_axis_mm2s_xi_tvalid,
    input  wire [  C_S_AXIS_XYZ_DATA_WIDTH-1:0] s_axis_mm2s_xi_tdata,
    input  wire [C_S_AXIS_XYZ_DATA_WIDTH/8-1:0] s_axis_mm2s_xi_tkeep,
    input  wire                                 s_axis_mm2s_xi_tlast,
    output wire                                 s_axis_mm2s_yi_tready,
    input  wire                                 s_axis_mm2s_yi_tvalid,
    input  wire [  C_S_AXIS_XYZ_DATA_WIDTH-1:0] s_axis_mm2s_yi_tdata,
    input  wire [C_S_AXIS_XYZ_DATA_WIDTH/8-1:0] s_axis_mm2s_yi_tkeep,
    input  wire                                 s_axis_mm2s_yi_tlast,
    input  wire                                 m_axis_s2mm_zo_tready,
    output wire                                 m_axis_s2mm_zo_tvalid,
    output wire [  C_S_AXIS_XYZ_DATA_WIDTH-1:0] m_axis_s2mm_zo_tdata,
    output wire [C_S_AXIS_XYZ_DATA_WIDTH/8-1:0] m_axis_s2mm_zo_tkeep,
    output wire                                 m_axis_s2mm_zo_tlast
);

    ///////////////////////////////////////////////////////////////////////////////
    // Begin RTL Design.
    ///////////////////////////////////////////////////////////////////////////////

    wire                                 pc_load_mode_sel;
    wire [                          1:0] pc_exec_mode_sel;
    wire [                          1:0] pc_store_mode_sel;
    wire [                        9-1:0] psu_fpv_depth;
    wire [                        9-1:0] xi_load_depth;
    wire [                        9-1:0] yi_load_depth;
    wire [                        5-1:0] xi_blk_num;
    wire [                        7-1:0] store_depth;
    wire [                        5-1:0] pc_fp_load_vec;
    wire                                 pc_fp_load_sel;
    wire                                 pc_fp_updt_en;
    wire                                 pc_fp_updt_sel;
    wire [                        5-1:0] pc_fp_updt_vec;
    wire [                        5-1:0] pc_fp0_exec_vec;
    wire [                        5-1:0] pc_fp1_exec_vec;
    wire                                 psu_acc;
    wire                                 psu_clr;
    wire                                 load_tile_sel;
    wire                                 exec_tile_sel;
    wire                                 pc_exec_start;
    wire                                 pc_store_start;
    wire                                 pc_loadx_done;
    wire                                 pc_loady_done;
    wire                                 pc_exec_done;
    wire                                 pc_store_done;
    wire [                          3:0] store_tapu_depth;
    wire [                          4:0] pc_quant_sf_exp;
    wire [                         15:0] pc_quant_sf_man;

    wire                                 i_axis_pccmd_tready;
    wire                                 i_axis_pccmd_tvalid;
    wire [C_S_AXIS_PCCMD_DATA_WIDTH-1:0] i_axis_pccmd_tdata;
    wire                                 i_axis_pcfbk_tready;
    wire                                 i_axis_pcfbk_tvalid;
    wire [C_S_AXIS_PCFBK_DATA_WIDTH-1:0] i_axis_pcfbk_tdata;
    wire                                 i_axis_mm2s_xi_tready;
    wire                                 i_axis_mm2s_xi_tvalid;
    wire [  C_S_AXIS_XYZ_DATA_WIDTH-1:0] i_axis_mm2s_xi_tdata;
    wire [C_S_AXIS_XYZ_DATA_WIDTH/8-1:0] i_axis_mm2s_xi_tkeep;
    wire                                 i_axis_mm2s_xi_tlast;
    wire                                 i_axis_mm2s_yi_tready;
    wire                                 i_axis_mm2s_yi_tvalid;
    wire [  C_S_AXIS_XYZ_DATA_WIDTH-1:0] i_axis_mm2s_yi_tdata;
    wire [C_S_AXIS_XYZ_DATA_WIDTH/8-1:0] i_axis_mm2s_yi_tkeep;
    wire                                 i_axis_mm2s_yi_tlast;
    wire                                 i_axis_s2mm_zo_tready;
    wire                                 i_axis_s2mm_zo_tvalid;
    wire [  C_S_AXIS_XYZ_DATA_WIDTH-1:0] i_axis_s2mm_zo_tdata;
    wire [C_S_AXIS_XYZ_DATA_WIDTH/8-1:0] i_axis_s2mm_zo_tkeep;
    wire                                 i_axis_s2mm_zo_tlast;

    // pccmd controller
    pccmd_ctrl pccmd_ctrl_inst (
        .clk                (ap_clk),
        .rst_n              (ap_resetn),
        .s_axis_pccmd_tvalid(i_axis_pccmd_tvalid),
        .s_axis_pccmd_tready(i_axis_pccmd_tready),
        .s_axis_pccmd_tdata (i_axis_pccmd_tdata),
        .m_axis_pcfbk_tvalid(i_axis_pcfbk_tvalid),
        .m_axis_pcfbk_tready(i_axis_pcfbk_tready),
        .m_axis_pcfbk_tdata (i_axis_pcfbk_tdata),
        .pc_load_mode_sel   (pc_load_mode_sel),
        .pc_exec_mode_sel   (pc_exec_mode_sel),
        .pc_store_mode_sel  (pc_store_mode_sel),
        .psu_fpv_depth      (psu_fpv_depth),
        .xi_load_depth      (xi_load_depth),
        .yi_load_depth      (yi_load_depth),
        .store_depth        (store_depth),
        .store_tapu_depth   (store_tapu_depth),
        .psu_acc            (psu_acc),
        .psu_clr            (psu_clr),
        .load_tile_sel      (load_tile_sel),
        .exec_tile_sel      (exec_tile_sel),
        .pc_fp_load_vec     (pc_fp_load_vec),
        .pc_fp_load_sel     (pc_fp_load_sel),
        .pc_fp_updt_en      (pc_fp_updt_en),
        .pc_fp_updt_sel     (pc_fp_updt_sel),
        .pc_fp_updt_vec     (pc_fp_updt_vec),
        .pc_fp0_exec_vec    (pc_fp0_exec_vec),
        .pc_fp1_exec_vec    (pc_fp1_exec_vec),
        .pc_quant_sf_exp    (pc_quant_sf_exp),
        .pc_quant_sf_man    (pc_quant_sf_man),
        .pc_exec_start      (pc_exec_start),
        .pc_store_start     (pc_store_start),
        .pc_loadx_done      (pc_loadx_done),
        .pc_loady_done      (pc_loady_done),
        .pc_exec_done       (pc_exec_done),
        .pc_store_done      (pc_store_done)
    );

    // processing core
    proc_core proc_core_inst (
        .clk                  (ap_clk),
        .rst_n                (ap_resetn),
        .pc_load_mode_sel     (pc_load_mode_sel),
        .pc_exec_mode_sel     (pc_exec_mode_sel),
        .pc_store_mode_sel    (pc_store_mode_sel),
        .psu_fpv_depth        (psu_fpv_depth),
        .xi_load_depth        (xi_load_depth),
        .yi_load_depth        (yi_load_depth),
        .store_depth          (store_depth),
        .store_tapu_depth     (store_tapu_depth),
        .psu_acc              (psu_acc),
        .psu_clr              (psu_clr),
        .load_tile_sel        (load_tile_sel),
        .exec_tile_sel        (exec_tile_sel),
        .pc_fp_load_vec       (pc_fp_load_vec),
        .pc_fp_load_sel       (pc_fp_load_sel),
        .pc_fp_updt_en        (pc_fp_updt_en),
        .pc_fp_updt_sel       (pc_fp_updt_sel),
        .pc_fp_updt_vec       (pc_fp_updt_vec),
        .pc_fp0_exec_vec      (pc_fp0_exec_vec),
        .pc_fp1_exec_vec      (pc_fp1_exec_vec),
        .pc_exec_start        (pc_exec_start),
        .pc_store_start       (pc_store_start),
        .pc_loadx_done        (pc_loadx_done),
        .pc_loady_done        (pc_loady_done),
        .pc_exec_done         (pc_exec_done),
        .pc_store_done        (pc_store_done),
        .sf_exp               (pc_quant_sf_exp),
        .sf_man               (pc_quant_sf_man),
        .s_axis_xi_load_tvalid(i_axis_mm2s_xi_tvalid),
        .s_axis_xi_load_tready(i_axis_mm2s_xi_tready),
        .s_axis_xi_load_tdata (i_axis_mm2s_xi_tdata),
        .s_axis_xi_load_tkeep (i_axis_mm2s_xi_tkeep),
        .s_axis_xi_load_tlast (i_axis_mm2s_xi_tlast),
        .s_axis_yi_load_tvalid(i_axis_mm2s_yi_tvalid),
        .s_axis_yi_load_tready(i_axis_mm2s_yi_tready),
        .s_axis_yi_load_tdata (i_axis_mm2s_yi_tdata),
        .s_axis_yi_load_tkeep (i_axis_mm2s_yi_tkeep),
        .s_axis_yi_load_tlast (i_axis_mm2s_yi_tlast),
        .m_axis_s2mm_tvalid   (i_axis_s2mm_zo_tvalid),
        .m_axis_s2mm_tready   (i_axis_s2mm_zo_tready),
        .m_axis_s2mm_tdata    (i_axis_s2mm_zo_tdata),
        .m_axis_s2mm_tkeep    (i_axis_s2mm_zo_tkeep),
        .m_axis_s2mm_tlast    (i_axis_s2mm_zo_tlast)
    );

    axis_register_256 axis_register_mm2s_xi (
        .aclk         (ap_clk),                 // input wire aclk
        .aresetn      (ap_resetn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mm2s_xi_tvalid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mm2s_xi_tready),  // output wire s_axis_tready
        .s_axis_tdata (s_axis_mm2s_xi_tdata),   // input wire [255 : 0] s_axis_tdata
        .s_axis_tkeep (s_axis_mm2s_xi_tkeep),   // input wire [31 : 0] s_axis_tkeep
        .s_axis_tlast (s_axis_mm2s_xi_tlast),   // input wire s_axis_tlast
        .m_axis_tvalid(i_axis_mm2s_xi_tvalid),  // output wire m_axis_tvalid
        .m_axis_tready(i_axis_mm2s_xi_tready),  // input wire m_axis_tready
        .m_axis_tdata (i_axis_mm2s_xi_tdata),   // output wire [255 : 0] m_axis_tdata
        .m_axis_tkeep (i_axis_mm2s_xi_tkeep),   // output wire [31 : 0] m_axis_tkeep
        .m_axis_tlast (i_axis_mm2s_xi_tlast)    // output wire m_axis_tlast
    );

    axis_register_256 axis_register_mm2s_yi (
        .aclk         (ap_clk),                 // input wire aclk
        .aresetn      (ap_resetn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mm2s_yi_tvalid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mm2s_yi_tready),  // output wire s_axis_tready
        .s_axis_tdata (s_axis_mm2s_yi_tdata),   // input wire [255 : 0] s_axis_tdata
        .s_axis_tkeep (s_axis_mm2s_yi_tkeep),   // input wire [31 : 0] s_axis_tkeep
        .s_axis_tlast (s_axis_mm2s_yi_tlast),   // input wire s_axis_tlast
        .m_axis_tvalid(i_axis_mm2s_yi_tvalid),  // output wire m_axis_tvalid
        .m_axis_tready(i_axis_mm2s_yi_tready),  // input wire m_axis_tready
        .m_axis_tdata (i_axis_mm2s_yi_tdata),   // output wire [255 : 0] m_axis_tdata
        .m_axis_tkeep (i_axis_mm2s_yi_tkeep),   // output wire [31 : 0] m_axis_tkeep
        .m_axis_tlast (i_axis_mm2s_yi_tlast)    // output wire m_axis_tlast
    );

    axis_register_256 axis_register_s2mm_zo (
        .aclk         (ap_clk),                 // input wire aclk
        .aresetn      (ap_resetn),              // input wire aresetn
        .s_axis_tvalid(i_axis_s2mm_zo_tvalid),  // input wire s_axis_tvalid
        .s_axis_tready(i_axis_s2mm_zo_tready),  // output wire s_axis_tready
        .s_axis_tdata (i_axis_s2mm_zo_tdata),   // input wire [255 : 0] s_axis_tdata
        .s_axis_tkeep (i_axis_s2mm_zo_tkeep),   // input wire [31 : 0] s_axis_tkeep
        .s_axis_tlast (i_axis_s2mm_zo_tlast),   // input wire s_axis_tlast
        .m_axis_tvalid(m_axis_s2mm_zo_tvalid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_s2mm_zo_tready),  // input wire m_axis_tready
        .m_axis_tdata (m_axis_s2mm_zo_tdata),   // output wire [255 : 0] m_axis_tdata
        .m_axis_tkeep (m_axis_s2mm_zo_tkeep),   // output wire [31 : 0] m_axis_tkeep
        .m_axis_tlast (m_axis_s2mm_zo_tlast)    // output wire m_axis_tlast
    );

    axis_register_32 axis_register_pccmd (
        .aclk         (ap_clk),               // input wire aclk
        .aresetn      (ap_resetn),            // input wire aresetn
        .s_axis_tvalid(s_axis_pccmd_tvalid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_pccmd_tready),  // output wire s_axis_tready
        .s_axis_tdata (s_axis_pccmd_tdata),   // input wire [31 : 0] s_axis_tdata
        .m_axis_tvalid(i_axis_pccmd_tvalid),  // output wire m_axis_tvalid
        .m_axis_tready(i_axis_pccmd_tready),  // input wire m_axis_tready
        .m_axis_tdata (i_axis_pccmd_tdata)    // output wire [31 : 0] m_axis_tdata
    );

    axis_register_8 axis_register_pcfbk (
        .aclk         (ap_clk),               // input wire aclk
        .aresetn      (ap_resetn),            // input wire aresetn
        .s_axis_tvalid(i_axis_pcfbk_tvalid),  // input wire s_axis_tvalid
        .s_axis_tready(i_axis_pcfbk_tready),  // output wire s_axis_tready
        .s_axis_tdata (i_axis_pcfbk_tdata),   // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_pcfbk_tvalid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_pcfbk_tready),  // input wire m_axis_tready
        .m_axis_tdata (m_axis_pcfbk_tdata)    // output wire [7 : 0] m_axis_tdata
    );

endmodule
