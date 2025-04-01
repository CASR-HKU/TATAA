// TODO: add license
`timescale 1 ns / 1 ps

// Top level of the kernel. Do not modify module name, parameters or ports.
module mem_kernel #(
    parameter integer C_S_AXI_CONTROL_ADDR_WIDTH = 8,
    parameter integer C_S_AXI_CONTROL_DATA_WIDTH = 32,
    parameter integer C_S_AXI_INSTR_ADDR_WIDTH   = 40,
    parameter integer C_S_AXI_INSTR_DATA_WIDTH   = 64,
    parameter integer C_S_AXI_CORE_ADDR_WIDTH    = 40,
    parameter integer C_S_AXI_CORE_DATA_WIDTH    = 256,
    parameter integer C_S_AXIS_XYZ_DATA_WIDTH    = 256,
    parameter integer C_S_AXIS_PCCMD_DATA_WIDTH  = 32,
    parameter integer C_S_AXIS_PCFBK_DATA_WIDTH  = 8
) (
    // System Signals
    input  wire                                    ap_clk,
    input  wire                                    ap_resetn,
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
    // AXI Control
    output wire                                    s_axi_control_awready,
    input  wire                                    s_axi_control_awvalid,
    input  wire [  C_S_AXI_CONTROL_ADDR_WIDTH-1:0] s_axi_control_awaddr,
    output wire                                    s_axi_control_wready,
    input  wire                                    s_axi_control_wvalid,
    input  wire [  C_S_AXI_CONTROL_DATA_WIDTH-1:0] s_axi_control_wdata,
    input  wire [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_control_wstrb,
    input  wire                                    s_axi_control_bready,
    output wire                                    s_axi_control_bvalid,
    output wire [                             1:0] s_axi_control_bresp,
    output wire                                    s_axi_control_arready,
    input  wire                                    s_axi_control_arvalid,
    input  wire [  C_S_AXI_CONTROL_ADDR_WIDTH-1:0] s_axi_control_araddr,
    input  wire                                    s_axi_control_rready,
    output wire                                    s_axi_control_rvalid,
    output wire [  C_S_AXI_CONTROL_DATA_WIDTH-1:0] s_axi_control_rdata,
    output wire [                             1:0] s_axi_control_rresp,
    // AXI channels for instruction
    input  wire                                    m_axi_instr_arready,
    output wire                                    m_axi_instr_arvalid,
    output wire [    C_S_AXI_INSTR_ADDR_WIDTH-1:0] m_axi_instr_araddr,
    output wire [                             7:0] m_axi_instr_arlen,
    output wire [                             2:0] m_axi_instr_arsize,
    output wire [                             1:0] m_axi_instr_arburst,
    output wire [                             2:0] m_axi_instr_arprot,
    output wire [                             3:0] m_axi_instr_arcache,
    output wire [                             3:0] m_axi_instr_aruser,
    output wire                                    m_axi_instr_rready,
    input  wire                                    m_axi_instr_rvalid,
    input  wire [    C_S_AXI_INSTR_DATA_WIDTH-1:0] m_axi_instr_rdata,
    input  wire [                             1:0] m_axi_instr_rresp,
    input  wire                                    m_axi_instr_rlast,
    // AXI channels for data xyz
    // m_axi for yizo
    input  wire                                    m_axi_yizo_awready,
    output wire                                    m_axi_yizo_awvalid,
    output wire [     C_S_AXI_CORE_ADDR_WIDTH-1:0] m_axi_yizo_awaddr,
    output wire [                             7:0] m_axi_yizo_awlen,
    output wire [                             2:0] m_axi_yizo_awsize,
    output wire [                             1:0] m_axi_yizo_awburst,
    output wire [                             2:0] m_axi_yizo_awprot,
    output wire [                             3:0] m_axi_yizo_awcache,
    output wire [                             3:0] m_axi_yizo_awuser,
    input  wire                                    m_axi_yizo_wready,
    output wire                                    m_axi_yizo_wvalid,
    output wire [     C_S_AXI_CORE_DATA_WIDTH-1:0] m_axi_yizo_wdata,
    output wire [   C_S_AXI_CORE_DATA_WIDTH/8-1:0] m_axi_yizo_wstrb,
    output wire                                    m_axi_yizo_wlast,
    output wire                                    m_axi_yizo_bready,
    input  wire                                    m_axi_yizo_bvalid,
    input  wire [                             1:0] m_axi_yizo_bresp,
    input  wire                                    m_axi_yizo_arready,
    output wire                                    m_axi_yizo_arvalid,
    output wire [     C_S_AXI_CORE_ADDR_WIDTH-1:0] m_axi_yizo_araddr,
    output wire [                             7:0] m_axi_yizo_arlen,
    output wire [                             2:0] m_axi_yizo_arsize,
    output wire [                             1:0] m_axi_yizo_arburst,
    output wire [                             2:0] m_axi_yizo_arprot,
    output wire [                             3:0] m_axi_yizo_arcache,
    output wire [                             3:0] m_axi_yizo_aruser,
    output wire                                    m_axi_yizo_rready,
    input  wire                                    m_axi_yizo_rvalid,
    input  wire [     C_S_AXI_CORE_DATA_WIDTH-1:0] m_axi_yizo_rdata,
    input  wire [                             1:0] m_axi_yizo_rresp,
    input  wire                                    m_axi_yizo_rlast,
    // m_axi for xi
    input  wire                                    m_axi_xi_arready,
    output wire                                    m_axi_xi_arvalid,
    output wire [     C_S_AXI_CORE_ADDR_WIDTH-1:0] m_axi_xi_araddr,
    output wire [                             7:0] m_axi_xi_arlen,
    output wire [                             2:0] m_axi_xi_arsize,
    output wire [                             1:0] m_axi_xi_arburst,
    output wire [                             2:0] m_axi_xi_arprot,
    output wire [                             3:0] m_axi_xi_arcache,
    output wire [                             3:0] m_axi_xi_aruser,
    output wire                                    m_axi_xi_rready,
    input  wire                                    m_axi_xi_rvalid,
    input  wire [     C_S_AXI_CORE_DATA_WIDTH-1:0] m_axi_xi_rdata,
    input  wire [                             1:0] m_axi_xi_rresp,
    input  wire                                    m_axi_xi_rlast,
    // AXI stream (master) for micro instructions to processing kernel
    input  wire                                    m_axis_pccmd_tready,
    output wire                                    m_axis_pccmd_tvalid,
    output wire [   C_S_AXIS_PCCMD_DATA_WIDTH-1:0] m_axis_pccmd_tdata,
    // AXI stream (slave) for feedback from processing kernel
    output wire                                    s_axis_pcfbk_tready,
    input  wire                                    s_axis_pcfbk_tvalid,
    input  wire [   C_S_AXIS_PCFBK_DATA_WIDTH-1:0] s_axis_pcfbk_tdata,
    // AXI stream (master) for data xyz to processing kernel
    input  wire                                    m_axis_mm2s_xi_tready,
    output wire                                    m_axis_mm2s_xi_tvalid,
    output wire [     C_S_AXIS_XYZ_DATA_WIDTH-1:0] m_axis_mm2s_xi_tdata,
    output wire [   C_S_AXIS_XYZ_DATA_WIDTH/8-1:0] m_axis_mm2s_xi_tkeep,
    output wire                                    m_axis_mm2s_xi_tlast,
    input  wire                                    m_axis_mm2s_yi_tready,
    output wire                                    m_axis_mm2s_yi_tvalid,
    output wire [     C_S_AXIS_XYZ_DATA_WIDTH-1:0] m_axis_mm2s_yi_tdata,
    output wire [   C_S_AXIS_XYZ_DATA_WIDTH/8-1:0] m_axis_mm2s_yi_tkeep,
    output wire                                    m_axis_mm2s_yi_tlast,
    output wire                                    s_axis_s2mm_zo_tready,
    input  wire                                    s_axis_s2mm_zo_tvalid,
    input  wire [     C_S_AXIS_XYZ_DATA_WIDTH-1:0] s_axis_s2mm_zo_tdata,
    input  wire [   C_S_AXIS_XYZ_DATA_WIDTH/8-1:0] s_axis_s2mm_zo_tkeep,
    input  wire                                    s_axis_s2mm_zo_tlast,
    output wire                                    interrupt
);

    ///////////////////////////////////////////////////////////////////////////////
    // Wires and Variables
    ///////////////////////////////////////////////////////////////////////////////
    (* DONT_TOUCH = "yes" *)wire        ap_start;
    (* DONT_TOUCH = "yes" *)wire        ap_idle;
    (* DONT_TOUCH = "yes" *)wire        ap_done;
    (* DONT_TOUCH = "yes" *)wire        ap_ready;
    (* DONT_TOUCH = "yes" *)wire        ap_idle_i;
    (* DONT_TOUCH = "yes" *)wire        ap_done_i;
    (* DONT_TOUCH = "yes" *)wire        ap_ready_i;
    (* DONT_TOUCH = "yes" *)wire        ap_idle_ii;
    (* DONT_TOUCH = "yes" *)wire        ap_done_ii;
    (* DONT_TOUCH = "yes" *)wire        ap_ready_ii;
    (* DONT_TOUCH = "yes" *)wire        ap_idle_iii;
    (* DONT_TOUCH = "yes" *)wire        ap_done_iii;
    (* DONT_TOUCH = "yes" *)wire        ap_ready_iii;
    (* DONT_TOUCH = "yes" *)wire        ap_idle_iv;
    (* DONT_TOUCH = "yes" *)wire        ap_done_iv;
    (* DONT_TOUCH = "yes" *)wire        ap_ready_iv;
    (* DONT_TOUCH = "yes" *)wire [63:0] instr_base_addr;
    (* DONT_TOUCH = "yes" *)wire [31:0] instr_btt;
    (* DONT_TOUCH = "yes" *)wire [63:0] yizo_base_addr;
    (* DONT_TOUCH = "yes" *)wire [63:0] xi_base_addr;
    (* DONT_TOUCH = "yes" *)wire [31:0] core_debug_status;
    (* DONT_TOUCH = "yes" *)wire [31:0] core_latency_cycles;
    (* DONT_TOUCH = "yes" *)wire [31:0] core_instr_status;
    (* DONT_TOUCH = "yes" *)wire [31:0] core_data_status;

    ///////////////////////////////////////////////////////////////////////////////
    // Begin control interface RTL.  Modifying not recommended.
    ///////////////////////////////////////////////////////////////////////////////

    ps_ctrl #(
        .PS_CTRL_AXI_ADDR_WIDTH(C_S_AXI_CONTROL_ADDR_WIDTH),
        .PS_CTRL_AXI_DATA_WIDTH(C_S_AXI_CONTROL_DATA_WIDTH)
    ) ps_ctrl_inst (
        .clk                  (ap_clk),
        .rst_n                (ap_resetn),
        .clk_en               (1'b1),
        .interrupt            (interrupt),
        .s_axi_control_awready(s_axi_control_awready),
        .s_axi_control_awvalid(s_axi_control_awvalid),
        .s_axi_control_awaddr (s_axi_control_awaddr),
        .s_axi_control_wready (s_axi_control_wready),
        .s_axi_control_wvalid (s_axi_control_wvalid),
        .s_axi_control_wdata  (s_axi_control_wdata),
        .s_axi_control_wstrb  (s_axi_control_wstrb),
        .s_axi_control_bready (s_axi_control_bready),
        .s_axi_control_bvalid (s_axi_control_bvalid),
        .s_axi_control_bresp  (s_axi_control_bresp),
        .s_axi_control_arready(s_axi_control_arready),
        .s_axi_control_arvalid(s_axi_control_arvalid),
        .s_axi_control_araddr (s_axi_control_araddr),
        .s_axi_control_rready (s_axi_control_rready),
        .s_axi_control_rvalid (s_axi_control_rvalid),
        .s_axi_control_rdata  (s_axi_control_rdata),
        .s_axi_control_rresp  (s_axi_control_rresp),
        .ap_start             (ap_start),
        .ap_done              (ap_done),
        .ap_idle              (ap_idle),
        .ap_ready             (ap_ready),
        .instr_base_addr      (instr_base_addr),
        .instr_btt            (instr_btt),
        .yizo_base_addr       (yizo_base_addr),
        .xi_base_addr         (xi_base_addr),
        .core_debug_status    (core_debug_status),
        .core_latency_cycles  (core_latency_cycles),
        .core_instr_status    (core_instr_status),
        .core_data_status     (core_data_status)
    );

    ///////////////////////////////////////////////////////////////////////////////
    // Begin RTL Design.
    ///////////////////////////////////////////////////////////////////////////////

    // instr loader
    wire                                i_instr_loader_tready;
    wire                                i_instr_loader_tvalid;
    wire [C_S_AXI_INSTR_DATA_WIDTH-1:0] i_instr_loader_tdata;
    instr_loader instr_loader_inst (
        .clk                (ap_clk),
        .rst_n              (ap_resetn),
        .ap_start           (ap_start),
        .ap_done            (ap_done),
        .ap_idle            (ap_idle),
        .ap_ready           (ap_ready),
        .instr_btt          (instr_btt),
        .base_addr          (instr_base_addr[39:0]),
        .instr_loader_status(core_instr_status),
        .m_axi_arready      (m_axi_instr_arready),
        .m_axi_arvalid      (m_axi_instr_arvalid),
        .m_axi_araddr       (m_axi_instr_araddr),
        .m_axi_arlen        (m_axi_instr_arlen),
        .m_axi_arsize       (m_axi_instr_arsize),
        .m_axi_arburst      (m_axi_instr_arburst),
        .m_axi_arprot       (m_axi_instr_arprot),
        .m_axi_arcache      (m_axi_instr_arcache),
        .m_axi_aruser       (m_axi_instr_aruser),
        .m_axi_rready       (m_axi_instr_rready),
        .m_axi_rvalid       (m_axi_instr_rvalid),
        .m_axi_rdata        (m_axi_instr_rdata),
        .m_axi_rresp        (m_axi_instr_rresp),
        .m_axi_rlast        (m_axi_instr_rlast),
        .m_instr_tready     (i_instr_loader_tready),
        .m_instr_tvalid     (i_instr_loader_tvalid),
        .m_instr_tdata      (i_instr_loader_tdata)
    );

    // data loader
    // input axis_mm2s_instr
    wire                                 i_axis_mm2s_cmd_xi_tready;
    wire                                 i_axis_mm2s_cmd_xi_tvalid;
    wire [                         87:0] i_axis_mm2s_cmd_xi_tdata;
    wire                                 i_axis_mm2s_cmd_yi_tready;
    wire                                 i_axis_mm2s_cmd_yi_tvalid;
    wire [                         87:0] i_axis_mm2s_cmd_yi_tdata;
    // input axis_s2mm_instr
    wire                                 i_axis_s2mm_cmd_zo_tready;
    wire                                 i_axis_s2mm_cmd_zo_tvalid;
    wire [                         87:0] i_axis_s2mm_cmd_zo_tdata;

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

    data_loader data_loader_inst (
        .clk                      (ap_clk),
        .rst_n                    (ap_resetn),
        .data_loader_status       (core_data_status),
        .m_axi_yizo_awready       (m_axi_yizo_awready),
        .m_axi_yizo_awvalid       (m_axi_yizo_awvalid),
        .m_axi_yizo_awaddr        (m_axi_yizo_awaddr),
        .m_axi_yizo_awlen         (m_axi_yizo_awlen),
        .m_axi_yizo_awsize        (m_axi_yizo_awsize),
        .m_axi_yizo_awburst       (m_axi_yizo_awburst),
        .m_axi_yizo_awprot        (m_axi_yizo_awprot),
        .m_axi_yizo_awcache       (m_axi_yizo_awcache),
        .m_axi_yizo_awuser        (m_axi_yizo_awuser),
        .m_axi_yizo_wready        (m_axi_yizo_wready),
        .m_axi_yizo_wvalid        (m_axi_yizo_wvalid),
        .m_axi_yizo_wdata         (m_axi_yizo_wdata),
        .m_axi_yizo_wstrb         (m_axi_yizo_wstrb),
        .m_axi_yizo_wlast         (m_axi_yizo_wlast),
        .m_axi_yizo_bready        (m_axi_yizo_bready),
        .m_axi_yizo_bvalid        (m_axi_yizo_bvalid),
        .m_axi_yizo_bresp         (m_axi_yizo_bresp),
        .m_axi_yizo_arready       (m_axi_yizo_arready),
        .m_axi_yizo_arvalid       (m_axi_yizo_arvalid),
        .m_axi_yizo_araddr        (m_axi_yizo_araddr),
        .m_axi_yizo_arlen         (m_axi_yizo_arlen),
        .m_axi_yizo_arsize        (m_axi_yizo_arsize),
        .m_axi_yizo_arburst       (m_axi_yizo_arburst),
        .m_axi_yizo_arprot        (m_axi_yizo_arprot),
        .m_axi_yizo_arcache       (m_axi_yizo_arcache),
        .m_axi_yizo_aruser        (m_axi_yizo_aruser),
        .m_axi_yizo_rready        (m_axi_yizo_rready),
        .m_axi_yizo_rvalid        (m_axi_yizo_rvalid),
        .m_axi_yizo_rdata         (m_axi_yizo_rdata),
        .m_axi_yizo_rresp         (m_axi_yizo_rresp),
        .m_axi_yizo_rlast         (m_axi_yizo_rlast),
        .m_axi_xi_arready         (m_axi_xi_arready),
        .m_axi_xi_arvalid         (m_axi_xi_arvalid),
        .m_axi_xi_araddr          (m_axi_xi_araddr),
        .m_axi_xi_arlen           (m_axi_xi_arlen),
        .m_axi_xi_arsize          (m_axi_xi_arsize),
        .m_axi_xi_arburst         (m_axi_xi_arburst),
        .m_axi_xi_arprot          (m_axi_xi_arprot),
        .m_axi_xi_arcache         (m_axi_xi_arcache),
        .m_axi_xi_aruser          (m_axi_xi_aruser),
        .m_axi_xi_rready          (m_axi_xi_rready),
        .m_axi_xi_rvalid          (m_axi_xi_rvalid),
        .m_axi_xi_rdata           (m_axi_xi_rdata),
        .m_axi_xi_rresp           (m_axi_xi_rresp),
        .m_axi_xi_rlast           (m_axi_xi_rlast),
        .s_axis_mm2s_cmd_xi_tready(i_axis_mm2s_cmd_xi_tready),
        .s_axis_mm2s_cmd_xi_tvalid(i_axis_mm2s_cmd_xi_tvalid),
        .s_axis_mm2s_cmd_xi_tdata (i_axis_mm2s_cmd_xi_tdata),
        .s_axis_mm2s_cmd_yi_tready(i_axis_mm2s_cmd_yi_tready),
        .s_axis_mm2s_cmd_yi_tvalid(i_axis_mm2s_cmd_yi_tvalid),
        .s_axis_mm2s_cmd_yi_tdata (i_axis_mm2s_cmd_yi_tdata),
        .s_axis_s2mm_cmd_zo_tready(i_axis_s2mm_cmd_zo_tready),
        .s_axis_s2mm_cmd_zo_tvalid(i_axis_s2mm_cmd_zo_tvalid),
        .s_axis_s2mm_cmd_zo_tdata (i_axis_s2mm_cmd_zo_tdata),
        .m_axis_mm2s_xi_tready    (i_axis_mm2s_xi_tready),
        .m_axis_mm2s_xi_tvalid    (i_axis_mm2s_xi_tvalid),
        .m_axis_mm2s_xi_tdata     (i_axis_mm2s_xi_tdata),
        .m_axis_mm2s_xi_tkeep     (i_axis_mm2s_xi_tkeep),
        .m_axis_mm2s_xi_tlast     (i_axis_mm2s_xi_tlast),
        .m_axis_mm2s_yi_tready    (i_axis_mm2s_yi_tready),
        .m_axis_mm2s_yi_tvalid    (i_axis_mm2s_yi_tvalid),
        .m_axis_mm2s_yi_tdata     (i_axis_mm2s_yi_tdata),
        .m_axis_mm2s_yi_tkeep     (i_axis_mm2s_yi_tkeep),
        .m_axis_mm2s_yi_tlast     (i_axis_mm2s_yi_tlast),
        .s_axis_s2mm_zo_tready    (i_axis_s2mm_zo_tready),
        .s_axis_s2mm_zo_tvalid    (i_axis_s2mm_zo_tvalid),
        .s_axis_s2mm_zo_tdata     (i_axis_s2mm_zo_tdata),
        .s_axis_s2mm_zo_tkeep     (i_axis_s2mm_zo_tkeep),
        .s_axis_s2mm_zo_tlast     (i_axis_s2mm_zo_tlast)
    );

    core_instr_ctrl core_instr_ctrl_inst (
        .clk                      (ap_clk),
        .rst_n                    (ap_resetn),
        .debug_status             (core_debug_status),
        .latency_cycles           (core_latency_cycles),
        .yizo_base_addr           (yizo_base_addr[39:0]),
        .xi_base_addr             (xi_base_addr[39:0]),
        .s_axis_instr_tready      (i_instr_loader_tready),
        .s_axis_instr_tvalid      (i_instr_loader_tvalid),
        .s_axis_instr_tdata       (i_instr_loader_tdata),
        .m_axis_mm2s_cmd_xi_tdata (i_axis_mm2s_cmd_xi_tdata),
        .m_axis_mm2s_cmd_xi_tvalid(i_axis_mm2s_cmd_xi_tvalid),
        .m_axis_mm2s_cmd_xi_tready(i_axis_mm2s_cmd_xi_tready),
        .m_axis_mm2s_cmd_yi_tdata (i_axis_mm2s_cmd_yi_tdata),
        .m_axis_mm2s_cmd_yi_tvalid(i_axis_mm2s_cmd_yi_tvalid),
        .m_axis_mm2s_cmd_yi_tready(i_axis_mm2s_cmd_yi_tready),
        .m_axis_s2mm_cmd_zo_tdata (i_axis_s2mm_cmd_zo_tdata),
        .m_axis_s2mm_cmd_zo_tvalid(i_axis_s2mm_cmd_zo_tvalid),
        .m_axis_s2mm_cmd_zo_tready(i_axis_s2mm_cmd_zo_tready),
        .m_axis_pccmd_tready      (i_axis_pccmd_tready),
        .m_axis_pccmd_tvalid      (i_axis_pccmd_tvalid),
        .m_axis_pccmd_tdata       (i_axis_pccmd_tdata),
        .s_axis_pcfbk_tready      (i_axis_pcfbk_tready),
        .s_axis_pcfbk_tvalid      (i_axis_pcfbk_tvalid),
        .s_axis_pcfbk_tdata       (i_axis_pcfbk_tdata)
    );

    axis_register_256 axis_register_mm2s_xi (
        .aclk         (ap_clk),                 // input wire aclk
        .aresetn      (ap_resetn),              // input wire aresetn
        .s_axis_tvalid(i_axis_mm2s_xi_tvalid),  // input wire s_axis_tvalid
        .s_axis_tready(i_axis_mm2s_xi_tready),  // output wire s_axis_tready
        .s_axis_tdata (i_axis_mm2s_xi_tdata),   // input wire [255 : 0] s_axis_tdata
        .s_axis_tkeep (i_axis_mm2s_xi_tkeep),   // input wire [31 : 0] s_axis_tkeep
        .s_axis_tlast (i_axis_mm2s_xi_tlast),   // input wire s_axis_tlast
        .m_axis_tvalid(m_axis_mm2s_xi_tvalid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mm2s_xi_tready),  // input wire m_axis_tready
        .m_axis_tdata (m_axis_mm2s_xi_tdata),   // output wire [255 : 0] m_axis_tdata
        .m_axis_tkeep (m_axis_mm2s_xi_tkeep),   // output wire [31 : 0] m_axis_tkeep
        .m_axis_tlast (m_axis_mm2s_xi_tlast)    // output wire m_axis_tlast
    );

    axis_register_256 axis_register_mm2s_yi (
        .aclk         (ap_clk),                 // input wire aclk
        .aresetn      (ap_resetn),              // input wire aresetn
        .s_axis_tvalid(i_axis_mm2s_yi_tvalid),  // input wire s_axis_tvalid
        .s_axis_tready(i_axis_mm2s_yi_tready),  // output wire s_axis_tready
        .s_axis_tdata (i_axis_mm2s_yi_tdata),   // input wire [255 : 0] s_axis_tdata
        .s_axis_tkeep (i_axis_mm2s_yi_tkeep),   // input wire [31 : 0] s_axis_tkeep
        .s_axis_tlast (i_axis_mm2s_yi_tlast),   // input wire s_axis_tlast
        .m_axis_tvalid(m_axis_mm2s_yi_tvalid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mm2s_yi_tready),  // input wire m_axis_tready
        .m_axis_tdata (m_axis_mm2s_yi_tdata),   // output wire [255 : 0] m_axis_tdata
        .m_axis_tkeep (m_axis_mm2s_yi_tkeep),   // output wire [31 : 0] m_axis_tkeep
        .m_axis_tlast (m_axis_mm2s_yi_tlast)    // output wire m_axis_tlast
    );

    axis_register_256 axis_register_s2mm_zo (
        .aclk         (ap_clk),                 // input wire aclk
        .aresetn      (ap_resetn),              // input wire aresetn
        .s_axis_tvalid(s_axis_s2mm_zo_tvalid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_s2mm_zo_tready),  // output wire s_axis_tready
        .s_axis_tdata (s_axis_s2mm_zo_tdata),   // input wire [255 : 0] s_axis_tdata
        .s_axis_tkeep (s_axis_s2mm_zo_tkeep),   // input wire [31 : 0] s_axis_tkeep
        .s_axis_tlast (s_axis_s2mm_zo_tlast),   // input wire s_axis_tlast
        .m_axis_tvalid(i_axis_s2mm_zo_tvalid),  // output wire m_axis_tvalid
        .m_axis_tready(i_axis_s2mm_zo_tready),  // input wire m_axis_tready
        .m_axis_tdata (i_axis_s2mm_zo_tdata),   // output wire [255 : 0] m_axis_tdata
        .m_axis_tkeep (i_axis_s2mm_zo_tkeep),   // output wire [31 : 0] m_axis_tkeep
        .m_axis_tlast (i_axis_s2mm_zo_tlast)    // output wire m_axis_tlast
    );

    axis_register_32 axis_register_pccmd (
        .aclk         (ap_clk),               // input wire aclk
        .aresetn      (ap_resetn),            // input wire aresetn
        .s_axis_tvalid(i_axis_pccmd_tvalid),  // input wire s_axis_tvalid
        .s_axis_tready(i_axis_pccmd_tready),  // output wire s_axis_tready
        .s_axis_tdata (i_axis_pccmd_tdata),   // input wire [31 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_pccmd_tvalid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_pccmd_tready),  // input wire m_axis_tready
        .m_axis_tdata (m_axis_pccmd_tdata)    // output wire [31 : 0] m_axis_tdata
    );

    axis_register_8 axis_register_pcfbk (
        .aclk         (ap_clk),               // input wire aclk
        .aresetn      (ap_resetn),            // input wire aresetn
        .s_axis_tvalid(s_axis_pcfbk_tvalid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_pcfbk_tready),  // output wire s_axis_tready
        .s_axis_tdata (s_axis_pcfbk_tdata),   // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(i_axis_pcfbk_tvalid),  // output wire m_axis_tvalid
        .m_axis_tready(i_axis_pcfbk_tready),  // input wire m_axis_tready
        .m_axis_tdata (i_axis_pcfbk_tdata)    // output wire [7 : 0] m_axis_tdata
    );

endmodule
