`timescale 1ns / 1ps

module instr_loader #(
    parameter AXI_ADDR_WIDTH   = 40,
    parameter AXI_DATA_WIDTH   = 64,
    parameter AXIS_INSTR_WIDTH = 64,
    parameter INSTR_FIFO_DEPTH = 256
) (
    // common signal
    input  logic                        clk,
    input  logic                        rst_n,
    // control signal
    input  logic                        ap_start,
    output logic                        ap_done,
    output logic                        ap_idle,
    output logic                        ap_ready,
    input  logic [                31:0] instr_btt,
    input  logic [  AXI_ADDR_WIDTH-1:0] base_addr,
    output logic [                31:0] instr_loader_status,
    // m_axi READ, AXI4 master
    input  logic                        m_axi_arready,
    output logic                        m_axi_arvalid,
    output logic [  AXI_ADDR_WIDTH-1:0] m_axi_araddr,
    output logic [                 7:0] m_axi_arlen,
    output logic [                 2:0] m_axi_arsize,
    output logic [                 1:0] m_axi_arburst,
    output logic [                 2:0] m_axi_arprot,
    output logic [                 3:0] m_axi_arcache,
    output logic [                 3:0] m_axi_aruser,
    output logic                        m_axi_rready,
    input  logic                        m_axi_rvalid,
    input  logic [  AXI_DATA_WIDTH-1:0] m_axi_rdata,
    input  logic [                 1:0] m_axi_rresp,
    input  logic                        m_axi_rlast,
    // m_axis, AXI4-Stream master
    input  logic                        m_instr_tready,
    output logic                        m_instr_tvalid,
    output logic [AXIS_INSTR_WIDTH-1:0] m_instr_tdata
);

    ///////////////////////////////////////////////////////////////////////////////
    // Local Parameters
    ///////////////////////////////////////////////////////////////////////////////
    localparam INSTR_FIFO_DATA_COUNT_WIDTH = $clog2(INSTR_FIFO_DEPTH) + 1;

    ///////////////////////////////////////////////////////////////////////////////
    // Wires and Variables
    ///////////////////////////////////////////////////////////////////////////////

    // Module status
    logic ap_start_r, ap_start_p, ap_busy;
    // AR channel
    logic                        m_axi_ar_hs;
    // R channel
    logic                        m_axi_r_hs;
    logic                        m_axi_r_done;
    // write to fifo
    logic [AXIS_INSTR_WIDTH-1:0] i_fifo_tdata;
    logic i_fifo_tready, i_fifo_tvalid, i_fifo_tlast;
    logic [         AXIS_INSTR_WIDTH/8-1:0] i_fifo_tkeep;
    // fifo
    logic [INSTR_FIFO_DATA_COUNT_WIDTH-1:0] instr_fifo_rd_cnt;
    logic [INSTR_FIFO_DATA_COUNT_WIDTH-1:0] instr_fifo_wr_cnt;
    // read from fifo
    logic m_instr_hs, m_instr_tlast;
    logic [AXIS_INSTR_WIDTH/8-1:0] m_instr_tkeep;

    always_comb begin
        ap_idle  = ~(ap_start | ap_busy);
        ap_ready = ap_done;
    end
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            ap_start_r <= 1'b0;
            ap_start_p <= 1'b0;
            ap_busy    <= 1'b0;
            ap_done    <= 1'b0;
        end else begin
            ap_start_r <= ap_start;
            ap_start_p <= &{~ap_start_p, ap_start, ~ap_start_r};  // a pulse on rising edge of ap_start
            ap_busy <= ap_done ? 1'b0 : ap_start ? 1'b1 : ap_busy;
            ap_done <= &{~ap_done, m_instr_hs & m_instr_tlast};
        end
    end

    assign m_axi_ar_hs = m_axi_arready & m_axi_arvalid;
    assign m_axi_r_hs  = m_axi_rvalid & m_axi_rready;
    assign m_instr_hs  = m_instr_tvalid & m_instr_tready;

    logic        i_axis_mm2s_cmd_tready;
    logic        i_axis_mm2s_cmd_tvalid;
    logic [87:0] i_axis_mm2s_cmd_tdata;
    logic [ 7:0] i_axis_mm2s_sts_tdata;
    logic        i_axis_mm2s_sts_tvalid;
    logic        i_axis_mm2s_sts_tready;
    logic        i_axis_mm2s_sts_tlast;
    logic [ 0:0] i_axis_mm2s_sts_tkeep;
    assign i_axis_mm2s_sts_tready = 1'b1;

    // for now, generate one cmd for all instructions
    logic cmd_tvalid_en, cmd_tvalid_r;
    assign cmd_tvalid_en = ap_start_p & ((~i_axis_mm2s_cmd_tvalid) | i_axis_mm2s_cmd_tready);
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            cmd_tvalid_r <= 1'b0;
        end else if (i_axis_mm2s_cmd_tvalid & i_axis_mm2s_cmd_tready) begin
            cmd_tvalid_r <= 1'b0;
        end else if (cmd_tvalid_en) begin
            cmd_tvalid_r <= 1'b1;
        end
    end
    assign i_axis_mm2s_cmd_tvalid = cmd_tvalid_r;

    logic [87:0] st_mm2s_cmd;
    assign st_mm2s_cmd[22:0]     = instr_btt[22:0];  // BTT
    assign st_mm2s_cmd[23]       = 1'b1;  // TYPE
    assign st_mm2s_cmd[29:24]    = 0;  // DSA
    assign st_mm2s_cmd[30]       = 1'b1;  // EOF
    assign st_mm2s_cmd[31]       = 1'b0;  // DRR
    assign st_mm2s_cmd[71:32]    = base_addr;  // SADDR
    assign st_mm2s_cmd[87:72]    = 0;

    assign i_axis_mm2s_cmd_tdata = st_mm2s_cmd;

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            instr_loader_status <= 0;
        end else begin
            instr_loader_status[31:24] <= m_axi_ar_hs ? instr_loader_status[31:24] + 1 : instr_loader_status[31:24];
            instr_loader_status[23:16] <= m_axi_r_hs ? instr_loader_status[23:16] + 1 : instr_loader_status[23:16];
            instr_loader_status[15:8]  <= m_instr_hs ? instr_loader_status[15:8] + 1 : instr_loader_status[15:8];
            instr_loader_status[7:0]   <= (m_axi_r_hs & (m_axi_rresp == 2'b00)) ? instr_loader_status[7:0] + 1 : instr_loader_status[7:0];
        end
    end


    logic mm2s_err;
    axi_datamover_instr instr_datamover (
        .m_axi_mm2s_aclk            (clk),  // input wire m_axi_mm2s_aclk
        .m_axi_mm2s_aresetn         (rst_n),  // input wire m_axi_mm2s_aresetn
        .mm2s_err                   (mm2s_err),  // output wire mm2s_err
        .m_axis_mm2s_cmdsts_aclk    (clk),  // input wire m_axis_mm2s_cmdsts_aclk
        .m_axis_mm2s_cmdsts_aresetn (rst_n),  // input wire m_axis_mm2s_cmdsts_aresetn
        .s_axis_mm2s_cmd_tvalid     (i_axis_mm2s_cmd_tvalid),  // input wire s_axis_mm2s_cmd_tvalid
        .s_axis_mm2s_cmd_tready     (i_axis_mm2s_cmd_tready),  // output wire s_axis_mm2s_cmd_tready
        .s_axis_mm2s_cmd_tdata      (i_axis_mm2s_cmd_tdata),  // input wire [79 : 0] s_axis_mm2s_cmd_tdata
        .m_axis_mm2s_sts_tvalid     (i_axis_mm2s_sts_tvalid),  // output wire m_axis_mm2s_sts_tvalid
        .m_axis_mm2s_sts_tready     (i_axis_mm2s_sts_tready),  // input wire m_axis_mm2s_sts_tready
        .m_axis_mm2s_sts_tdata      (i_axis_mm2s_sts_tdata),  // output wire [7 : 0] m_axis_mm2s_sts_tdata
        .m_axis_mm2s_sts_tkeep      (i_axis_mm2s_sts_tkeep),  // output wire [0 : 0] m_axis_mm2s_sts_tkeep
        .m_axis_mm2s_sts_tlast      (i_axis_mm2s_sts_tlast),  // output wire m_axis_mm2s_sts_tlast
        .m_axi_mm2s_araddr          (m_axi_araddr),  // output wire [31 : 0] m_axi_mm2s_araddr
        .m_axi_mm2s_arlen           (m_axi_arlen),  // output wire [7 : 0] m_axi_mm2s_arlen
        .m_axi_mm2s_arsize          (m_axi_arsize),  // output wire [2 : 0] m_axi_mm2s_arsize
        .m_axi_mm2s_arburst         (m_axi_arburst),  // output wire [1 : 0] m_axi_mm2s_arburst
        .m_axi_mm2s_arprot          (m_axi_arprot),  // output wire [2 : 0] m_axi_mm2s_arprot
        .m_axi_mm2s_arcache         (m_axi_arcache),  // output wire [3 : 0] m_axi_mm2s_arcache
        .m_axi_mm2s_aruser          (m_axi_aruser),  // output wire [3 : 0] m_axi_mm2s_aruser
        .m_axi_mm2s_arvalid         (m_axi_arvalid),  // output wire m_axi_mm2s_arvalid
        .m_axi_mm2s_arready         (m_axi_arready),  // input wire m_axi_mm2s_arready
        .m_axi_mm2s_rdata           (m_axi_rdata),  // input wire [127 : 0] m_axi_mm2s_rdata
        .m_axi_mm2s_rresp           (m_axi_rresp),  // input wire [1 : 0] m_axi_mm2s_rresp
        .m_axi_mm2s_rlast           (m_axi_rlast),  // input wire m_axi_mm2s_rlast
        .m_axi_mm2s_rvalid          (m_axi_rvalid),  // input wire m_axi_mm2s_rvalid
        .m_axi_mm2s_rready          (m_axi_rready),  // output wire m_axi_mm2s_rready
        .m_axis_mm2s_tdata          (i_fifo_tdata),  // output wire [63 : 0] m_axis_mm2s_tdata
        .m_axis_mm2s_tkeep          (i_fifo_tkeep),  // output wire [7 : 0] m_axis_mm2s_tkeep
        .m_axis_mm2s_tlast          (i_fifo_tlast),  // output wire m_axis_mm2s_tlast
        .m_axis_mm2s_tvalid         (i_fifo_tvalid),  // output wire m_axis_mm2s_tvalid
        .m_axis_mm2s_tready         (i_fifo_tready)  // input wire m_axis_mm2s_tready
    );
    /**************/
    /* INSTR FIFO */
    /**************/

    fifo_axis #(
        .FIFO_AXIS_DEPTH      (INSTR_FIFO_DEPTH),
        .FIFO_AXIS_TDATA_WIDTH(AXIS_INSTR_WIDTH),
        .FIFO_DATA_COUNT_WIDTH(INSTR_FIFO_DATA_COUNT_WIDTH),
        .FIFO_ADV_FEATURES    ("1000")
    ) instr_fifo (
        //common signal
        .clk          (clk),
        .rst_n        (rst_n),
        // s_axis
        .s_axis_tready(i_fifo_tready),
        .s_axis_tvalid(i_fifo_tvalid),
        .s_axis_tdata (i_fifo_tdata),
        .s_axis_tkeep (i_fifo_tkeep),
        .s_axis_tlast (i_fifo_tlast),
        // m_axis
        .m_axis_tready(m_instr_tready),
        .m_axis_tvalid(m_instr_tvalid),
        .m_axis_tdata (m_instr_tdata),
        .m_axis_tkeep (m_instr_tkeep),
        .m_axis_tlast (m_instr_tlast)
    );

endmodule
