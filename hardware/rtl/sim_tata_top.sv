`timescale 1ns / 1ps

// simulation test for processing core
module sim_tata_top (
    // common signals
    input  logic        clk,
    input  logic        rst_n,
    // AXI Control
    output logic        s_axi_control_awready,
    input  logic        s_axi_control_awvalid,
    input  logic [ 7:0] s_axi_control_awaddr,
    output logic        s_axi_control_wready,
    input  logic        s_axi_control_wvalid,
    input  logic [31:0] s_axi_control_wdata,
    input  logic [ 3:0] s_axi_control_wstrb,
    input  logic        s_axi_control_bready,
    output logic        s_axi_control_bvalid,
    output logic [ 1:0] s_axi_control_bresp,
    output logic        s_axi_control_arready,
    input  logic        s_axi_control_arvalid,
    input  logic [ 7:0] s_axi_control_araddr,
    input  logic        s_axi_control_rready,
    output logic        s_axi_control_rvalid,
    output logic [31:0] s_axi_control_rdata,
    output logic [ 1:0] s_axi_control_rresp
);

    logic         m_axi_instr_arready;
    logic         m_axi_instr_arvalid;
    logic [ 39:0] m_axi_instr_araddr;
    logic [  7:0] m_axi_instr_arlen;
    logic [  2:0] m_axi_instr_arsize;
    logic [  1:0] m_axi_instr_arburst;
    logic [  2:0] m_axi_instr_arprot;
    logic [  3:0] m_axi_instr_arcache;
    logic [  3:0] m_axi_instr_aruser;
    logic         m_axi_instr_rready;
    logic         m_axi_instr_rvalid;
    logic [ 63:0] m_axi_instr_rdata;
    logic [  1:0] m_axi_instr_rresp;
    logic         m_axi_instr_rlast;

    logic         m_axi_yizo_awready;
    logic         m_axi_yizo_awvalid;
    logic [ 39:0] m_axi_yizo_awaddr;
    logic [  7:0] m_axi_yizo_awlen;
    logic [  2:0] m_axi_yizo_awsize;
    logic [  1:0] m_axi_yizo_awburst;
    logic [  2:0] m_axi_yizo_awprot;
    logic [  3:0] m_axi_yizo_awcache;
    logic [  3:0] m_axi_yizo_awuser;
    logic         m_axi_yizo_wready;
    logic         m_axi_yizo_wvalid;
    logic [255:0] m_axi_yizo_wdata;
    logic [ 31:0] m_axi_yizo_wstrb;
    logic         m_axi_yizo_wlast;
    logic         m_axi_yizo_bready;
    logic         m_axi_yizo_bvalid;
    logic [  1:0] m_axi_yizo_bresp;
    logic         m_axi_yizo_arready;
    logic         m_axi_yizo_arvalid;
    logic [ 39:0] m_axi_yizo_araddr;
    logic [  7:0] m_axi_yizo_arlen;
    logic [  2:0] m_axi_yizo_arsize;
    logic [  1:0] m_axi_yizo_arburst;
    logic [  2:0] m_axi_yizo_arprot;
    logic [  3:0] m_axi_yizo_arcache;
    logic [  3:0] m_axi_yizo_aruser;
    logic         m_axi_yizo_rready;
    logic         m_axi_yizo_rvalid;
    logic [255:0] m_axi_yizo_rdata;
    logic [  1:0] m_axi_yizo_rresp;
    logic         m_axi_yizo_rlast;

    logic         m_axi_xi_arready;
    logic         m_axi_xi_arvalid;
    logic [ 39:0] m_axi_xi_araddr;
    logic [  7:0] m_axi_xi_arlen;
    logic [  2:0] m_axi_xi_arsize;
    logic [  1:0] m_axi_xi_arburst;
    logic [  2:0] m_axi_xi_arprot;
    logic [  3:0] m_axi_xi_arcache;
    logic [  3:0] m_axi_xi_aruser;
    logic         m_axi_xi_rready;
    logic         m_axi_xi_rvalid;
    logic [255:0] m_axi_xi_rdata;
    logic [  1:0] m_axi_xi_rresp;
    logic         m_axi_xi_rlast;

    tata_top_wrapper tata_top_wrapper_sim_inst (
        .ap_clk               (clk),
        .ap_resetn            (rst_n),
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
        .m_axi_instr_arready  (m_axi_instr_arready),
        .m_axi_instr_arvalid  (m_axi_instr_arvalid),
        .m_axi_instr_araddr   (m_axi_instr_araddr),
        .m_axi_instr_arlen    (m_axi_instr_arlen),
        .m_axi_instr_arsize   (m_axi_instr_arsize),
        .m_axi_instr_arburst  (m_axi_instr_arburst),
        .m_axi_instr_arprot   (m_axi_instr_arprot),
        .m_axi_instr_arcache  (m_axi_instr_arcache),
        .m_axi_instr_aruser   (m_axi_instr_aruser),
        .m_axi_instr_rready   (m_axi_instr_rready),
        .m_axi_instr_rvalid   (m_axi_instr_rvalid),
        .m_axi_instr_rdata    (m_axi_instr_rdata),
        .m_axi_instr_rresp    (m_axi_instr_rresp),
        .m_axi_instr_rlast    (m_axi_instr_rlast),
        .m_axi_yizo_awready   (m_axi_yizo_awready),
        .m_axi_yizo_awvalid   (m_axi_yizo_awvalid),
        .m_axi_yizo_awaddr    (m_axi_yizo_awaddr),
        .m_axi_yizo_awlen     (m_axi_yizo_awlen),
        .m_axi_yizo_awsize    (m_axi_yizo_awsize),
        .m_axi_yizo_awburst   (m_axi_yizo_awburst),
        .m_axi_yizo_awprot    (m_axi_yizo_awprot),
        .m_axi_yizo_awcache   (m_axi_yizo_awcache),
        .m_axi_yizo_awuser    (m_axi_yizo_awuser),
        .m_axi_yizo_wready    (m_axi_yizo_wready),
        .m_axi_yizo_wvalid    (m_axi_yizo_wvalid),
        .m_axi_yizo_wdata     (m_axi_yizo_wdata),
        .m_axi_yizo_wstrb     (m_axi_yizo_wstrb),
        .m_axi_yizo_wlast     (m_axi_yizo_wlast),
        .m_axi_yizo_bready    (m_axi_yizo_bready),
        .m_axi_yizo_bvalid    (m_axi_yizo_bvalid),
        .m_axi_yizo_bresp     (m_axi_yizo_bresp),
        .m_axi_yizo_arready   (m_axi_yizo_arready),
        .m_axi_yizo_arvalid   (m_axi_yizo_arvalid),
        .m_axi_yizo_araddr    (m_axi_yizo_araddr),
        .m_axi_yizo_arlen     (m_axi_yizo_arlen),
        .m_axi_yizo_arsize    (m_axi_yizo_arsize),
        .m_axi_yizo_arburst   (m_axi_yizo_arburst),
        .m_axi_yizo_arprot    (m_axi_yizo_arprot),
        .m_axi_yizo_arcache   (m_axi_yizo_arcache),
        .m_axi_yizo_aruser    (m_axi_yizo_aruser),
        .m_axi_yizo_rready    (m_axi_yizo_rready),
        .m_axi_yizo_rvalid    (m_axi_yizo_rvalid),
        .m_axi_yizo_rdata     (m_axi_yizo_rdata),
        .m_axi_yizo_rresp     (m_axi_yizo_rresp),
        .m_axi_yizo_rlast     (m_axi_yizo_rlast),
        .m_axi_xi_arready     (m_axi_xi_arready),
        .m_axi_xi_arvalid     (m_axi_xi_arvalid),
        .m_axi_xi_araddr      (m_axi_xi_araddr),
        .m_axi_xi_arlen       (m_axi_xi_arlen),
        .m_axi_xi_arsize      (m_axi_xi_arsize),
        .m_axi_xi_arburst     (m_axi_xi_arburst),
        .m_axi_xi_arprot      (m_axi_xi_arprot),
        .m_axi_xi_arcache     (m_axi_xi_arcache),
        .m_axi_xi_aruser      (m_axi_xi_aruser),
        .m_axi_xi_rready      (m_axi_xi_rready),
        .m_axi_xi_rvalid      (m_axi_xi_rvalid),
        .m_axi_xi_rdata       (m_axi_xi_rdata),
        .m_axi_xi_rresp       (m_axi_xi_rresp),
        .m_axi_xi_rlast       (m_axi_xi_rlast)
    );

    logic rsta_busy_instr, rstb_busy_instr;
    blk_mem_instr blk_mem_sim_instr_inst (
        .rsta_busy    (rsta_busy_instr),      // output wire rsta_busy
        .rstb_busy    (rstb_busy_instr),      // output wire rstb_busy
        .s_aclk       (clk),                  // input wire s_aclk
        .s_aresetn    (rst_n),                // input wire s_aresetn
        .s_axi_arid   (1'b0),                 // input wire [0 : 0] s_axi_arid
        .s_axi_araddr (m_axi_instr_araddr),   // input wire [31 : 0] s_axi_araddr
        .s_axi_arlen  (m_axi_instr_arlen),    // input wire [7 : 0] s_axi_arlen
        .s_axi_arsize (m_axi_instr_arsize),   // input wire [2 : 0] s_axi_arsize
        .s_axi_arburst(m_axi_instr_arburst),  // input wire [1 : 0] s_axi_arburst
        .s_axi_arvalid(m_axi_instr_arvalid),  // input wire s_axi_arvalid
        .s_axi_arready(m_axi_instr_arready),  // output wire s_axi_arready
        .s_axi_rdata  (m_axi_instr_rdata),    // output wire [63 : 0] s_axi_rdata
        .s_axi_rresp  (m_axi_instr_rresp),    // output wire [1 : 0] s_axi_rresp
        .s_axi_rlast  (m_axi_instr_rlast),    // output wire s_axi_rlast
        .s_axi_rvalid (m_axi_instr_rvalid),   // output wire s_axi_rvalid
        .s_axi_rready (m_axi_instr_rready)    // input wire s_axi_rready
    );

    logic rsta_busy_yizo, rstb_busy_yizo;
    blk_mem_data blk_mem_sim_yizo_inst (
        .rsta_busy    (rsta_busy_yizo),     // output wire rsta_busy
        .rstb_busy    (rstb_busy_yizo),     // output wire rstb_busy
        .s_aclk       (clk),                // input wire s_aclk
        .s_aresetn    (rst_n),              // input wire s_aresetn
        .s_axi_awid   (1'b0),               // input wire [0 : 0] s_axi_awid
        .s_axi_awaddr (m_axi_yizo_awaddr),   // input wire [31 : 0] s_axi_awaddr
        .s_axi_awlen  (m_axi_yizo_awlen),    // input wire [7 : 0] s_axi_awlen
        .s_axi_awsize (m_axi_yizo_awsize),   // input wire [2 : 0] s_axi_awsize
        .s_axi_awburst(m_axi_yizo_awburst),  // input wire [1 : 0] s_axi_awburst
        .s_axi_awvalid(m_axi_yizo_awvalid),  // input wire s_axi_awvalid
        .s_axi_awready(m_axi_yizo_awready),  // output wire s_axi_awready
        .s_axi_wdata  (m_axi_yizo_wdata),    // input wire [127 : 0] s_axi_wdata
        .s_axi_wstrb  (m_axi_yizo_wstrb),    // input wire [15 : 0] s_axi_wstrb
        .s_axi_wlast  (m_axi_yizo_wlast),    // input wire s_axi_wlast
        .s_axi_wvalid (m_axi_yizo_wvalid),   // input wire s_axi_wvalid
        .s_axi_wready (m_axi_yizo_wready),   // output wire s_axi_wready
        .s_axi_bresp  (m_axi_yizo_bresp),    // output wire [1 : 0] s_axi_bresp
        .s_axi_bvalid (m_axi_yizo_bvalid),   // output wire s_axi_bvalid
        .s_axi_bready (m_axi_yizo_bready),   // input wire s_axi_bready
        .s_axi_arid   (1'b0),               // input wire [0 : 0] s_axi_arid
        .s_axi_araddr (m_axi_yizo_araddr),   // input wire [31 : 0] s_axi_araddr
        .s_axi_arlen  (m_axi_yizo_arlen),    // input wire [7 : 0] s_axi_arlen
        .s_axi_arsize (m_axi_yizo_arsize),   // input wire [2 : 0] s_axi_arsize
        .s_axi_arburst(m_axi_yizo_arburst),  // input wire [1 : 0] s_axi_arburst
        .s_axi_arvalid(m_axi_yizo_arvalid),  // input wire s_axi_arvalid
        .s_axi_arready(m_axi_yizo_arready),  // output wire s_axi_arready
        .s_axi_rdata  (m_axi_yizo_rdata),    // output wire [127 : 0] s_axi_rdata
        .s_axi_rresp  (m_axi_yizo_rresp),    // output wire [1 : 0] s_axi_rresp
        .s_axi_rlast  (m_axi_yizo_rlast),    // output wire s_axi_rlast
        .s_axi_rvalid (m_axi_yizo_rvalid),   // output wire s_axi_rvalid
        .s_axi_rready (m_axi_yizo_rready)    // input wire s_axi_rready
    );

    logic rsta_busy_xi, rstb_busy_xi;
    blk_mem_data blk_mem_sim_xyz_inst (
        .rsta_busy    (rsta_busy_xi),     // output wire rsta_busy
        .rstb_busy    (rstb_busy_xi),     // output wire rstb_busy
        .s_aclk       (clk),                // input wire s_aclk
        .s_aresetn    (rst_n),              // input wire s_aresetn
        .s_axi_awid   (1'b0),               // input wire [0 : 0] s_axi_awid
        .s_axi_awaddr (m_axi_xi_awaddr),   // input wire [31 : 0] s_axi_awaddr
        .s_axi_awlen  (m_axi_xi_awlen),    // input wire [7 : 0] s_axi_awlen
        .s_axi_awsize (m_axi_xi_awsize),   // input wire [2 : 0] s_axi_awsize
        .s_axi_awburst(m_axi_xi_awburst),  // input wire [1 : 0] s_axi_awburst
        .s_axi_awvalid(m_axi_xi_awvalid),  // input wire s_axi_awvalid
        .s_axi_awready(m_axi_xi_awready),  // output wire s_axi_awready
        .s_axi_wdata  (m_axi_xi_wdata),    // input wire [127 : 0] s_axi_wdata
        .s_axi_wstrb  (m_axi_xi_wstrb),    // input wire [15 : 0] s_axi_wstrb
        .s_axi_wlast  (m_axi_xi_wlast),    // input wire s_axi_wlast
        .s_axi_wvalid (m_axi_xi_wvalid),   // input wire s_axi_wvalid
        .s_axi_wready (m_axi_xi_wready),   // output wire s_axi_wready
        .s_axi_bresp  (m_axi_xi_bresp),    // output wire [1 : 0] s_axi_bresp
        .s_axi_bvalid (m_axi_xi_bvalid),   // output wire s_axi_bvalid
        .s_axi_bready (m_axi_xi_bready),   // input wire s_axi_bready
        .s_axi_arid   (1'b0),               // input wire [0 : 0] s_axi_arid
        .s_axi_araddr (m_axi_xi_araddr),   // input wire [31 : 0] s_axi_araddr
        .s_axi_arlen  (m_axi_xi_arlen),    // input wire [7 : 0] s_axi_arlen
        .s_axi_arsize (m_axi_xi_arsize),   // input wire [2 : 0] s_axi_arsize
        .s_axi_arburst(m_axi_xi_arburst),  // input wire [1 : 0] s_axi_arburst
        .s_axi_arvalid(m_axi_xi_arvalid),  // input wire s_axi_arvalid
        .s_axi_arready(m_axi_xi_arready),  // output wire s_axi_arready
        .s_axi_rdata  (m_axi_xi_rdata),    // output wire [127 : 0] s_axi_rdata
        .s_axi_rresp  (m_axi_xi_rresp),    // output wire [1 : 0] s_axi_rresp
        .s_axi_rlast  (m_axi_xi_rlast),    // output wire s_axi_rlast
        .s_axi_rvalid (m_axi_xi_rvalid),   // output wire s_axi_rvalid
        .s_axi_rready (m_axi_xi_rready)    // input wire s_axi_rready
    );

endmodule
