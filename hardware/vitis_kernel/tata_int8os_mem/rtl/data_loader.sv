`timescale 1ns / 1ps

module data_loader #(
    parameter AXI_ADDR_WIDTH  = 40,
    parameter CORE_STS_WIDTH  = 8,
    parameter AXIS_DATA_WIDTH = 256
) (
    // common signal
    input  logic                         clk,
    input  logic                         rst_n,
    output logic [                 31:0] data_loader_status,
    // m_axi for yizo
    input  logic                         m_axi_yizo_awready,
    output logic                         m_axi_yizo_awvalid,
    output logic [   AXI_ADDR_WIDTH-1:0] m_axi_yizo_awaddr,
    output logic [                  7:0] m_axi_yizo_awlen,
    output logic [                  2:0] m_axi_yizo_awsize,
    output logic [                  1:0] m_axi_yizo_awburst,
    output logic [                  2:0] m_axi_yizo_awprot,
    output logic [                  3:0] m_axi_yizo_awcache,
    output logic [                  3:0] m_axi_yizo_awuser,
    input  logic                         m_axi_yizo_wready,
    output logic                         m_axi_yizo_wvalid,
    output logic [  AXIS_DATA_WIDTH-1:0] m_axi_yizo_wdata,
    output logic [AXIS_DATA_WIDTH/8-1:0] m_axi_yizo_wstrb,
    output logic                         m_axi_yizo_wlast,
    output logic                         m_axi_yizo_bready,
    input  logic                         m_axi_yizo_bvalid,
    input  logic [                  1:0] m_axi_yizo_bresp,
    input  logic                         m_axi_yizo_arready,
    output logic                         m_axi_yizo_arvalid,
    output logic [   AXI_ADDR_WIDTH-1:0] m_axi_yizo_araddr,
    output logic [                  7:0] m_axi_yizo_arlen,
    output logic [                  2:0] m_axi_yizo_arsize,
    output logic [                  1:0] m_axi_yizo_arburst,
    output logic [                  2:0] m_axi_yizo_arprot,
    output logic [                  3:0] m_axi_yizo_arcache,
    output logic [                  3:0] m_axi_yizo_aruser,
    output logic                         m_axi_yizo_rready,
    input  logic                         m_axi_yizo_rvalid,
    input  logic [  AXIS_DATA_WIDTH-1:0] m_axi_yizo_rdata,
    input  logic [                  1:0] m_axi_yizo_rresp,
    input  logic                         m_axi_yizo_rlast,
    // m_axi for xi
    input  logic                         m_axi_xi_arready,
    output logic                         m_axi_xi_arvalid,
    output logic [   AXI_ADDR_WIDTH-1:0] m_axi_xi_araddr,
    output logic [                  7:0] m_axi_xi_arlen,
    output logic [                  2:0] m_axi_xi_arsize,
    output logic [                  1:0] m_axi_xi_arburst,
    output logic [                  2:0] m_axi_xi_arprot,
    output logic [                  3:0] m_axi_xi_arcache,
    output logic [                  3:0] m_axi_xi_aruser,
    output logic                         m_axi_xi_rready,
    input  logic                         m_axi_xi_rvalid,
    input  logic [  AXIS_DATA_WIDTH-1:0] m_axi_xi_rdata,
    input  logic [                  1:0] m_axi_xi_rresp,
    input  logic                         m_axi_xi_rlast,
    // input axis_mm2s_cmd
    output logic                         s_axis_mm2s_cmd_xi_tready,
    input  logic                         s_axis_mm2s_cmd_xi_tvalid,
    input  logic [                 87:0] s_axis_mm2s_cmd_xi_tdata,
    output logic                         s_axis_mm2s_cmd_yi_tready,
    input  logic                         s_axis_mm2s_cmd_yi_tvalid,
    input  logic [                 87:0] s_axis_mm2s_cmd_yi_tdata,
    // input axis_s2mm_cmd
    output logic                         s_axis_s2mm_cmd_zo_tready,
    input  logic                         s_axis_s2mm_cmd_zo_tvalid,
    input  logic [                 87:0] s_axis_s2mm_cmd_zo_tdata,
    // output axis_mm2s_xi
    input  logic                         m_axis_mm2s_xi_tready,
    output logic                         m_axis_mm2s_xi_tvalid,
    output logic [  AXIS_DATA_WIDTH-1:0] m_axis_mm2s_xi_tdata,
    output logic [AXIS_DATA_WIDTH/8-1:0] m_axis_mm2s_xi_tkeep,
    output logic                         m_axis_mm2s_xi_tlast,
    // output axis_mm2s_yi
    input  logic                         m_axis_mm2s_yi_tready,
    output logic                         m_axis_mm2s_yi_tvalid,
    output logic [  AXIS_DATA_WIDTH-1:0] m_axis_mm2s_yi_tdata,
    output logic [AXIS_DATA_WIDTH/8-1:0] m_axis_mm2s_yi_tkeep,
    output logic                         m_axis_mm2s_yi_tlast,
    // input axis_s2mm
    output logic                         s_axis_s2mm_zo_tready,
    input  logic                         s_axis_s2mm_zo_tvalid,
    input  logic [  AXIS_DATA_WIDTH-1:0] s_axis_s2mm_zo_tdata,
    input  logic [AXIS_DATA_WIDTH/8-1:0] s_axis_s2mm_zo_tkeep,
    input  logic                         s_axis_s2mm_zo_tlast
);

    // mm2s_sts_xy
    logic                         i_axis_mm2s_sts_xi_tready;
    logic                         i_axis_mm2s_sts_xi_tvalid;
    logic [   CORE_STS_WIDTH-1:0] i_axis_mm2s_sts_xi_tdata;
    logic                         i_axis_mm2s_sts_xi_tkeep;
    logic                         i_axis_mm2s_sts_xi_tlast;
    logic                         i_axis_mm2s_sts_yi_tready;
    logic                         i_axis_mm2s_sts_yi_tvalid;
    logic [   CORE_STS_WIDTH-1:0] i_axis_mm2s_sts_yi_tdata;
    logic                         i_axis_mm2s_sts_yi_tkeep;
    logic                         i_axis_mm2s_sts_yi_tlast;
    // mm2s_sts_zo
    logic                         i_axis_s2mm_sts_zo_tready;
    logic                         i_axis_s2mm_sts_zo_tvalid;
    logic [   CORE_STS_WIDTH-1:0] i_axis_s2mm_sts_zo_tdata;
    logic                         i_axis_s2mm_sts_zo_tkeep;
    logic                         i_axis_s2mm_sts_zo_tlast;

    assign i_axis_mm2s_sts_xi_tready = 1'b1;
    assign i_axis_mm2s_sts_yi_tready = 1'b1;
    assign i_axis_s2mm_sts_zo_tready = 1'b1;

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            data_loader_status <= 0;
        end else begin
            data_loader_status[7:0] <= (i_axis_s2mm_sts_zo_tready & i_axis_s2mm_sts_zo_tvalid & (i_axis_s2mm_sts_zo_tdata != 8'h80)) ? i_axis_s2mm_sts_zo_tdata : data_loader_status[7:0];
            data_loader_status[15:8] <= (i_axis_s2mm_sts_zo_tready & i_axis_s2mm_sts_zo_tvalid & (i_axis_s2mm_sts_zo_tdata != 8'h80)) ? data_loader_status[15:8] + 1 : data_loader_status[15:8];
            data_loader_status[23:16] <= (i_axis_mm2s_sts_xi_tready & i_axis_mm2s_sts_xi_tvalid & (i_axis_mm2s_sts_xi_tdata != 8'h80)) ? data_loader_status[23:16] + 1 : data_loader_status[23:16];
            data_loader_status[31:24] <= (i_axis_mm2s_sts_yi_tready & i_axis_mm2s_sts_yi_tvalid & (i_axis_mm2s_sts_yi_tdata != 8'h80)) ? data_loader_status[31:24] + 1 : data_loader_status[31:24];
        end
    end

    logic mm2s_err_xi, mm2s_err_yi, s2mm_err_zo;
    axi_datamover_yizo axi_datamover_yizo_inst (
        .m_axi_mm2s_aclk            (clk),  // input wire m_axi_mm2s_aclk
        .m_axi_mm2s_aresetn         (rst_n),  // input wire m_axi_mm2s_aresetn
        .mm2s_err                   (mm2s_err_yi),  // output wire mm2s_err
        .m_axis_mm2s_cmdsts_aclk    (clk),  // input wire m_axis_mm2s_cmdsts_aclk
        .m_axis_mm2s_cmdsts_aresetn (rst_n),  // input wire m_axis_mm2s_cmdsts_aresetn
        .s_axis_mm2s_cmd_tvalid     (s_axis_mm2s_cmd_yi_tvalid),  // input wire s_axis_mm2s_cmd_tvalid
        .s_axis_mm2s_cmd_tready     (s_axis_mm2s_cmd_yi_tready),  // output wire s_axis_mm2s_cmd_tready
        .s_axis_mm2s_cmd_tdata      (s_axis_mm2s_cmd_yi_tdata),            // input wire [79 : 0] s_axis_mm2s_cmd_tdata
        .m_axis_mm2s_sts_tvalid     (i_axis_mm2s_sts_yi_tvalid),  // output wire m_axis_mm2s_sts_tvalid
        .m_axis_mm2s_sts_tready     (i_axis_mm2s_sts_yi_tready),  // input wire m_axis_mm2s_sts_tready
        .m_axis_mm2s_sts_tdata      (i_axis_mm2s_sts_yi_tdata),            // output wire [7 : 0] m_axis_mm2s_sts_tdata
        .m_axis_mm2s_sts_tkeep      (i_axis_mm2s_sts_yi_tkeep),            // output wire [0 : 0] m_axis_mm2s_sts_tkeep
        .m_axis_mm2s_sts_tlast      (i_axis_mm2s_sts_yi_tlast),  // output wire m_axis_mm2s_sts_tlast
        .m_axi_mm2s_araddr          (m_axi_yizo_araddr),  // output wire [31 : 0] m_axi_mm2s_araddr
        .m_axi_mm2s_arlen           (m_axi_yizo_arlen),  // output wire [7 : 0] m_axi_mm2s_arlen
        .m_axi_mm2s_arsize          (m_axi_yizo_arsize),  // output wire [2 : 0] m_axi_mm2s_arsize
        .m_axi_mm2s_arburst         (m_axi_yizo_arburst),  // output wire [1 : 0] m_axi_mm2s_arburst
        .m_axi_mm2s_arprot          (m_axi_yizo_arprot),  // output wire [2 : 0] m_axi_mm2s_arprot
        .m_axi_mm2s_arcache         (m_axi_yizo_arcache),  // output wire [3 : 0] m_axi_mm2s_arcache
        .m_axi_mm2s_aruser          (m_axi_yizo_aruser),  // output wire [3 : 0] m_axi_mm2s_aruser
        .m_axi_mm2s_arvalid         (m_axi_yizo_arvalid),  // output wire m_axi_mm2s_arvalid
        .m_axi_mm2s_arready         (m_axi_yizo_arready),  // input wire m_axi_mm2s_arready
        .m_axi_mm2s_rdata           (m_axi_yizo_rdata),  // input wire [63 : 0] m_axi_mm2s_rdata
        .m_axi_mm2s_rresp           (m_axi_yizo_rresp),  // input wire [1 : 0] m_axi_mm2s_rresp
        .m_axi_mm2s_rlast           (m_axi_yizo_rlast),  // input wire m_axi_mm2s_rlast
        .m_axi_mm2s_rvalid          (m_axi_yizo_rvalid),  // input wire m_axi_mm2s_rvalid
        .m_axi_mm2s_rready          (m_axi_yizo_rready),  // output wire m_axi_mm2s_rready
        .m_axis_mm2s_tdata          (m_axis_mm2s_yi_tdata),  // output wire [63 : 0] m_axis_mm2s_tdata
        .m_axis_mm2s_tkeep          (m_axis_mm2s_yi_tkeep),  // output wire [7 : 0] m_axis_mm2s_tkeep
        .m_axis_mm2s_tlast          (m_axis_mm2s_yi_tlast),  // output wire m_axis_mm2s_tlast
        .m_axis_mm2s_tvalid         (m_axis_mm2s_yi_tvalid),  // output wire m_axis_mm2s_tvalid
        .m_axis_mm2s_tready         (m_axis_mm2s_yi_tready),  // input wire m_axis_mm2s_tready
        .m_axi_s2mm_aclk            (clk),  // input wire m_axi_s2mm_aclk
        .m_axi_s2mm_aresetn         (rst_n),  // input wire m_axi_s2mm_aresetn
        .s2mm_err                   (s2mm_err_zo),  // output wire s2mm_err
        .m_axis_s2mm_cmdsts_awclk   (clk),  // input wire m_axis_s2mm_cmdsts_awclk
        .m_axis_s2mm_cmdsts_aresetn (rst_n),  // input wire m_axis_s2mm_cmdsts_aresetn
        .s_axis_s2mm_cmd_tvalid     (s_axis_s2mm_cmd_zo_tvalid),  // input wire s_axis_s2mm_cmd_tvalid
        .s_axis_s2mm_cmd_tready     (s_axis_s2mm_cmd_zo_tready),  // output wire s_axis_s2mm_cmd_tready
        .s_axis_s2mm_cmd_tdata      (s_axis_s2mm_cmd_zo_tdata),            // input wire [79 : 0] s_axis_s2mm_cmd_tdata
        .m_axis_s2mm_sts_tvalid     (i_axis_s2mm_sts_zo_tvalid),  // output wire m_axis_s2mm_sts_tvalid
        .m_axis_s2mm_sts_tready     (i_axis_s2mm_sts_zo_tready),  // input wire m_axis_s2mm_sts_tready
        .m_axis_s2mm_sts_tdata      (i_axis_s2mm_sts_zo_tdata),            // output wire [7 : 0] m_axis_s2mm_sts_tdata
        .m_axis_s2mm_sts_tkeep      (i_axis_s2mm_sts_zo_tkeep),            // output wire [0 : 0] m_axis_s2mm_sts_tkeep
        .m_axis_s2mm_sts_tlast      (i_axis_s2mm_sts_zo_tlast),  // output wire m_axis_s2mm_sts_tlast
        .m_axi_s2mm_awaddr          (m_axi_yizo_awaddr),  // output wire [31 : 0] m_axi_s2mm_awaddr
        .m_axi_s2mm_awlen           (m_axi_yizo_awlen),  // output wire [7 : 0] m_axi_s2mm_awlen
        .m_axi_s2mm_awsize          (m_axi_yizo_awsize),  // output wire [2 : 0] m_axi_s2mm_awsize
        .m_axi_s2mm_awburst         (m_axi_yizo_awburst),  // output wire [1 : 0] m_axi_s2mm_awburst
        .m_axi_s2mm_awprot          (m_axi_yizo_awprot),  // output wire [2 : 0] m_axi_s2mm_awprot
        .m_axi_s2mm_awcache         (m_axi_yizo_awcache),  // output wire [3 : 0] m_axi_s2mm_awcache
        .m_axi_s2mm_awuser          (m_axi_yizo_awuser),  // output wire [3 : 0] m_axi_s2mm_awuser
        .m_axi_s2mm_awvalid         (m_axi_yizo_awvalid),  // output wire m_axi_s2mm_awvalid
        .m_axi_s2mm_awready         (m_axi_yizo_awready),  // input wire m_axi_s2mm_awready
        .m_axi_s2mm_wdata           (m_axi_yizo_wdata),  // output wire [63 : 0] m_axi_s2mm_wdata
        .m_axi_s2mm_wstrb           (m_axi_yizo_wstrb),  // output wire [7 : 0] m_axi_s2mm_wstrb
        .m_axi_s2mm_wlast           (m_axi_yizo_wlast),  // output wire m_axi_s2mm_wlast
        .m_axi_s2mm_wvalid          (m_axi_yizo_wvalid),  // output wire m_axi_s2mm_wvalid
        .m_axi_s2mm_wready          (m_axi_yizo_wready),  // input wire m_axi_s2mm_wready
        .m_axi_s2mm_bresp           (m_axi_yizo_bresp),  // input wire [1 : 0] m_axi_s2mm_bresp
        .m_axi_s2mm_bvalid          (m_axi_yizo_bvalid),  // input wire m_axi_s2mm_bvalid
        .m_axi_s2mm_bready          (m_axi_yizo_bready),  // output wire m_axi_s2mm_bready
        .s_axis_s2mm_tdata          (s_axis_s2mm_zo_tdata),  // input wire [63 : 0] s_axis_s2mm_tdata
        .s_axis_s2mm_tkeep          (s_axis_s2mm_zo_tkeep),  // input wire [7 : 0] s_axis_s2mm_tkeep
        .s_axis_s2mm_tlast          (s_axis_s2mm_zo_tlast),  // input wire s_axis_s2mm_tlast
        .s_axis_s2mm_tvalid         (s_axis_s2mm_zo_tvalid),  // input wire s_axis_s2mm_tvalid
        .s_axis_s2mm_tready         (s_axis_s2mm_zo_tready)  // output wire s_axis_s2mm_tready
    );

    axi_datamover_xi axi_datamover_xi_inst (
        .m_axi_mm2s_aclk            (clk),  // input wire m_axi_mm2s_aclk
        .m_axi_mm2s_aresetn         (rst_n),  // input wire m_axi_mm2s_aresetn
        .mm2s_err                   (mm2s_err_xi),  // output wire mm2s_err
        .m_axis_mm2s_cmdsts_aclk    (clk),  // input wire m_axis_mm2s_cmdsts_aclk
        .m_axis_mm2s_cmdsts_aresetn (rst_n),  // input wire m_axis_mm2s_cmdsts_aresetn
        .s_axis_mm2s_cmd_tvalid     (s_axis_mm2s_cmd_xi_tvalid),  // input wire s_axis_mm2s_cmd_tvalid
        .s_axis_mm2s_cmd_tready     (s_axis_mm2s_cmd_xi_tready),  // output wire s_axis_mm2s_cmd_tready
        .s_axis_mm2s_cmd_tdata      (s_axis_mm2s_cmd_xi_tdata),            // input wire [79 : 0] s_axis_mm2s_cmd_tdata
        .m_axis_mm2s_sts_tvalid     (i_axis_mm2s_sts_xi_tvalid),  // output wire m_axis_mm2s_sts_tvalid
        .m_axis_mm2s_sts_tready     (i_axis_mm2s_sts_xi_tready),  // input wire m_axis_mm2s_sts_tready
        .m_axis_mm2s_sts_tdata      (i_axis_mm2s_sts_xi_tdata),            // output wire [7 : 0] m_axis_mm2s_sts_tdata
        .m_axis_mm2s_sts_tkeep      (i_axis_mm2s_sts_xi_tkeep),            // output wire [0 : 0] m_axis_mm2s_sts_tkeep
        .m_axis_mm2s_sts_tlast      (i_axis_mm2s_sts_xi_tlast),  // output wire m_axis_mm2s_sts_tlast
        .m_axi_mm2s_araddr          (m_axi_xi_araddr),  // output wire [31 : 0] m_axi_mm2s_araddr
        .m_axi_mm2s_arlen           (m_axi_xi_arlen),  // output wire [7 : 0] m_axi_mm2s_arlen
        .m_axi_mm2s_arsize          (m_axi_xi_arsize),  // output wire [2 : 0] m_axi_mm2s_arsize
        .m_axi_mm2s_arburst         (m_axi_xi_arburst),  // output wire [1 : 0] m_axi_mm2s_arburst
        .m_axi_mm2s_arprot          (m_axi_xi_arprot),  // output wire [2 : 0] m_axi_mm2s_arprot
        .m_axi_mm2s_arcache         (m_axi_xi_arcache),  // output wire [3 : 0] m_axi_mm2s_arcache
        .m_axi_mm2s_aruser          (m_axi_xi_aruser),  // output wire [3 : 0] m_axi_mm2s_aruser
        .m_axi_mm2s_arvalid         (m_axi_xi_arvalid),  // output wire m_axi_mm2s_arvalid
        .m_axi_mm2s_arready         (m_axi_xi_arready),  // input wire m_axi_mm2s_arready
        .m_axi_mm2s_rdata           (m_axi_xi_rdata),  // input wire [63 : 0] m_axi_mm2s_rdata
        .m_axi_mm2s_rresp           (m_axi_xi_rresp),  // input wire [1 : 0] m_axi_mm2s_rresp
        .m_axi_mm2s_rlast           (m_axi_xi_rlast),  // input wire m_axi_mm2s_rlast
        .m_axi_mm2s_rvalid          (m_axi_xi_rvalid),  // input wire m_axi_mm2s_rvalid
        .m_axi_mm2s_rready          (m_axi_xi_rready),  // output wire m_axi_mm2s_rready
        .m_axis_mm2s_tdata          (m_axis_mm2s_xi_tdata),  // output wire [63 : 0] m_axis_mm2s_tdata
        .m_axis_mm2s_tkeep          (m_axis_mm2s_xi_tkeep),  // output wire [7 : 0] m_axis_mm2s_tkeep
        .m_axis_mm2s_tlast          (m_axis_mm2s_xi_tlast),  // output wire m_axis_mm2s_tlast
        .m_axis_mm2s_tvalid         (m_axis_mm2s_xi_tvalid),  // output wire m_axis_mm2s_tvalid
        .m_axis_mm2s_tready         (m_axis_mm2s_xi_tready)  // input wire m_axis_mm2s_tready
    );

endmodule
