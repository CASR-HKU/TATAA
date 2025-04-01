`timescale 1ns / 1ps

module proc_core #(
    // TAPU
    parameter LEFT_WIDTH            = 8,
    parameter RIGHT_WIDTH           = 8,
    parameter TOP_WIDTH             = 48,
    parameter BOTTOM_WIDTH          = 48,
    parameter PRELD_WIDTH           = 16,
    parameter ROWS                  = 4,
    parameter COLS                  = 16,
    parameter PSU_FIFO_DEPTH        = 512,
    parameter PSU_DATA_WIDTH        = 32,
    // DMRFX & Y
    parameter AXIS_LOAD_DATA_WIDTH  = 256,
    parameter TAPU_NUM              = 8,
    parameter X_SYS_DATA_WIDTH      = 8,
    parameter X_SYS_PORT_NUM        = 32,
    parameter X_BRAM_SIZE           = 262144,
    parameter X_BRAM_ADDR_WIDTH     = 10,
    parameter X_BRAM_DATA_WIDTH     = 256,
    parameter X_LOAD_ADDR_WIDTH     = 9,
    parameter X_EXEC_ADDR_WIDTH     = 9,
    parameter Y_SYS_DATA_WIDTH      = 32,
    parameter Y_SYS_PORT_NUM        = 16,
    parameter Y_BRAM_SIZE           = 8192,
    parameter Y_BRAM_ADDR_WIDTH     = 5,
    parameter Y_BRAM_DATA_WIDTH     = 256,
    parameter Y_LOAD_ADDR_WIDTH     = 5,
    parameter Y_EXEC_ADDR_WIDTH     = 5,
    // Output Z
    parameter AXIS_S2MM_DATA_WIDTH  = 256,
    // Controller
    parameter RC_WIDTH              = 2,
    parameter PSU_DEPTH_WIDTH       = 9,
    parameter DMRFX_ADDR_WIDTH      = 9,
    parameter DMRFY_ADDR_WIDTH      = 5,
    parameter EXP_BUF_ADDR_WIDTH    = 5,
    parameter BLK_NUM_WIDTH         = 5,
    parameter EXTRA_LATENCY         = 51,
    // FP MUL and FP ADD
    parameter FPM_DATA_WIDTH        = 16,
    parameter BFLOAT_LAT_CYCLE      = 16,
    // Dual Mode Buffer (small)
    parameter DMB_FIFO_DEPTH        = 16,
    parameter DMB_FIFO_DATA_WIDTH   = 512,
    // Scaling & quantization
    parameter QUANT_SCALE_EXP_WIDTH = 5,
    parameter QUANT_SCALE_MAN_WIDTH = 16,
    parameter QUANT_OUT_WIDTH       = 8
) (
    // common signals
    input  logic                              clk,
    input  logic                              rst_n,
    // parameters from command controller
    input  logic                              pc_load_mode_sel,
    input  logic [                       1:0] pc_exec_mode_sel,
    input  logic [                       1:0] pc_store_mode_sel,
    input  logic [       PSU_DEPTH_WIDTH-1:0] psu_fpv_depth,
    input  logic [       PSU_DEPTH_WIDTH-1:0] xi_load_depth,
    input  logic [       PSU_DEPTH_WIDTH-1:0] yi_load_depth,
    input  logic [                       6:0] store_depth,
    input  logic [                       3:0] store_tapu_depth,
    input  logic                              psu_clr,
    input  logic                              psu_acc,
    input  logic                              load_tile_sel,
    input  logic                              exec_tile_sel,
    // fp mode
    input  logic [     Y_LOAD_ADDR_WIDTH-1:0] pc_fp_load_vec,
    input  logic                              pc_fp_load_sel,
    input  logic                              pc_fp_updt_en,
    input  logic                              pc_fp_updt_sel,
    input  logic [     Y_LOAD_ADDR_WIDTH-1:0] pc_fp_updt_vec,
    input  logic [     Y_LOAD_ADDR_WIDTH-1:0] pc_fp0_exec_vec,
    input  logic [     Y_LOAD_ADDR_WIDTH-1:0] pc_fp1_exec_vec,
    // control signals
    input  logic                              pc_exec_start,
    input  logic                              pc_store_start,
    output logic                              pc_loadx_done,
    output logic                              pc_loady_done,
    output logic                              pc_exec_done,
    output logic                              pc_store_done,
    // scaling factors
    input  logic [ QUANT_SCALE_EXP_WIDTH-1:0] sf_exp,
    input  logic [ QUANT_SCALE_MAN_WIDTH-1:0] sf_man,
    // input x data stream
    input  logic                              s_axis_xi_load_tvalid,
    output logic                              s_axis_xi_load_tready,
    input  logic [  AXIS_LOAD_DATA_WIDTH-1:0] s_axis_xi_load_tdata,
    input  logic [AXIS_LOAD_DATA_WIDTH/8-1:0] s_axis_xi_load_tkeep,
    input  logic                              s_axis_xi_load_tlast,
    // input y data stream
    input  logic                              s_axis_yi_load_tvalid,
    output logic                              s_axis_yi_load_tready,
    input  logic [  AXIS_LOAD_DATA_WIDTH-1:0] s_axis_yi_load_tdata,
    input  logic [AXIS_LOAD_DATA_WIDTH/8-1:0] s_axis_yi_load_tkeep,
    input  logic                              s_axis_yi_load_tlast,
    // output z data (s2mm) stream
    output logic                              m_axis_s2mm_tvalid,
    input  logic                              m_axis_s2mm_tready,
    output logic [  AXIS_S2MM_DATA_WIDTH-1:0] m_axis_s2mm_tdata,
    output logic [AXIS_S2MM_DATA_WIDTH/8-1:0] m_axis_s2mm_tkeep,
    output logic                              m_axis_s2mm_tlast
);

    /****************************************** DMRFX ******************************************/
    logic [           AXIS_LOAD_DATA_WIDTH-1:0] i_axis_dmrfx_load_tdata;
    logic                                       i_axis_dmrfx_load_tvalid;
    logic                                       i_axis_dmrfx_load_tready;
    logic [         AXIS_LOAD_DATA_WIDTH/8-1:0] i_axis_dmrfx_load_tkeep;
    logic                                       i_axis_dmrfx_load_tlast;
    logic                                       dmrfx_load_tile_sel;
    logic [              X_LOAD_ADDR_WIDTH-1:0] dmrfx_exec_addr;
    logic                                       dmrfx_exec_tile_sel;
    logic                                       dmrfx_load_done;
    logic [              X_LOAD_ADDR_WIDTH-1:0] dmrfx_load_depth;
    logic [X_SYS_DATA_WIDTH*X_SYS_PORT_NUM-1:0] dmrfx_exec_data;

    dmrf_x #(
        .AXIS_LOAD_DATA_WIDTH(AXIS_LOAD_DATA_WIDTH),
        .TAPU_NUM            (TAPU_NUM),
        .SYS_DATA_WIDTH      (X_SYS_DATA_WIDTH),
        .SYS_PORT_NUM        (X_SYS_PORT_NUM),
        .BRAM_SIZE           (X_BRAM_SIZE),
        .BRAM_ADDR_WIDTH     (X_BRAM_ADDR_WIDTH),
        .BRAM_DATA_WIDTH     (X_BRAM_DATA_WIDTH),
        .LOAD_ADDR_WIDTH     (X_LOAD_ADDR_WIDTH),
        .EXEC_ADDR_WIDTH     (X_EXEC_ADDR_WIDTH)
    ) dmrf_x_inst (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .s_axis_dmrfx_load_tdata (i_axis_dmrfx_load_tdata),
        .s_axis_dmrfx_load_tvalid(i_axis_dmrfx_load_tvalid),
        .s_axis_dmrfx_load_tready(i_axis_dmrfx_load_tready),
        .s_axis_dmrfx_load_tkeep (i_axis_dmrfx_load_tkeep),
        .s_axis_dmrfx_load_tlast (i_axis_dmrfx_load_tlast),
        .dmrfx_load_tile_sel     (dmrfx_load_tile_sel),
        .dmrfx_exec_addr         (dmrfx_exec_addr),
        .dmrfx_exec_tile_sel     (dmrfx_exec_tile_sel),
        .dmrfx_load_done         (dmrfx_load_done),
        .dmrfx_load_depth        (dmrfx_load_depth),
        .dmrfx_exec_data         (dmrfx_exec_data)
    );

    always_ff @(posedge clk) begin
        dmrfx_load_tile_sel <= load_tile_sel;
        dmrfx_exec_tile_sel <= exec_tile_sel;
        dmrfx_load_depth    <= xi_load_depth;
    end
    /****************************************** DMRFX ******************************************/

    /****************************************** DMRFY ******************************************/
    // TODO: parameterize the TAPU_NUM
    logic [                  7:0][                              255:0] i_axis_dmrfy_load_tdata;
    logic [                  7:0][                              255:0] dmrfy_fp_updt_data;
    logic [                  7:0]                                      i_axis_dmrfy_load_tvalid;
    logic [                  7:0]                                      i_axis_dmrfy_load_tready;
    logic [                  7:0][                               31:0] i_axis_dmrfy_load_tkeep;
    logic [                  7:0]                                      i_axis_dmrfy_load_tlast;
    logic [                  7:0][              Y_LOAD_ADDR_WIDTH-1:0] dmrfy_load_depth;
    logic [                  7:0]                                      dmrfy_mode_sel;
    logic [                  7:0]                                      dmrfy_load_tile_sel;
    logic [                  7:0][              Y_EXEC_ADDR_WIDTH-1:0] dmrfy_exec_addr;
    logic [                  7:0]                                      dmrfy_load_done;
    logic [                  7:0][Y_SYS_DATA_WIDTH*Y_SYS_PORT_NUM-1:0] dmrfy_exec_data;
    logic [                  7:0]                                      dmrfy_fp_updt_en;
    logic [                  7:0]                                      dmrfy_fp_updt_sel;
    logic [                  7:0][              Y_LOAD_ADDR_WIDTH-1:0] dmrfy_fp0_updt_addr;
    logic [                  7:0][              Y_LOAD_ADDR_WIDTH-1:0] dmrfy_fp0_exec_addr;
    logic [                  7:0][              Y_LOAD_ADDR_WIDTH-1:0] dmrfy_fp1_updt_addr;
    logic [                  7:0][              Y_LOAD_ADDR_WIDTH-1:0] dmrfy_fp1_exec_addr;

    logic                                                              fp_updt_en_pipeout;
    logic                                                              fp_wmem_en_pipeout;
    logic                                                              fp_updt_sel_pipeout;
    logic [Y_LOAD_ADDR_WIDTH-1:0]                                      fp_updt_addr_pipeout;
    logic                                                              fp_exec_start;

    logic [X_LOAD_ADDR_WIDTH-1:0]                                      dmrfy_int8_load_depth;
    logic [X_LOAD_ADDR_WIDTH-1:0]                                      dmrfy_int8_exec_addr;

    assign fp_exec_start = pc_exec_start & (pc_exec_mode_sel[1] == 1'b1);

    delay_chain #(
        .DW (1),
        .LEN(BFLOAT_LAT_CYCLE)
    ) delay_chain_fp_updten (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (fp_exec_start & pc_fp_updt_en),
        .out  (fp_updt_en_pipeout)
    );

    delay_chain #(
        .DW (1),
        .LEN(BFLOAT_LAT_CYCLE)
    ) delay_chain_fp_wmemen (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (fp_exec_start & (~pc_fp_updt_en)),
        .out  (fp_wmem_en_pipeout)
    );

    delay_chain #(
        .DW (1),
        .LEN(BFLOAT_LAT_CYCLE)
    ) delay_chain_fp_updtsel (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (pc_fp_updt_sel),
        .out  (fp_updt_sel_pipeout)
    );

    delay_chain #(
        .DW (Y_LOAD_ADDR_WIDTH),
        .LEN(BFLOAT_LAT_CYCLE)
    ) delay_chain_fp_updaddr (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (pc_fp_updt_vec),
        .out  (fp_updt_addr_pipeout)
    );

    // for TAPU0, the dmrf_y should have the same size with dmrf_x
    dmrf_y #(
        .AXIS_LOAD_DATA_WIDTH(AXIS_LOAD_DATA_WIDTH),
        .SYS_DATA_WIDTH      (Y_SYS_DATA_WIDTH),
        .SYS_PORT_NUM        (Y_SYS_PORT_NUM),
        .BRAM_SIZE           (X_BRAM_SIZE),
        .BRAM_ADDR_WIDTH     (X_BRAM_ADDR_WIDTH),
        .BRAM_DATA_WIDTH     (X_BRAM_DATA_WIDTH),
        .LOAD_ADDR_WIDTH     (X_LOAD_ADDR_WIDTH),
        .EXEC_ADDR_WIDTH     (X_EXEC_ADDR_WIDTH)
    ) dmrf_y_int8_inst (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .s_axis_dmrfy_load_tdata (i_axis_dmrfy_load_tdata[0]),
        .s_axis_dmrfy_load_tvalid(i_axis_dmrfy_load_tvalid[0]),
        .s_axis_dmrfy_load_tready(i_axis_dmrfy_load_tready[0]),
        .s_axis_dmrfy_load_tkeep (i_axis_dmrfy_load_tkeep[0]),
        .s_axis_dmrfy_load_tlast (i_axis_dmrfy_load_tlast[0]),
        .dmrfy_load_depth        (dmrfy_int8_load_depth),
        .dmrfy_mode_sel          (dmrfy_mode_sel[0]),
        .dmrfy_load_tile_sel     (dmrfy_load_tile_sel[0]),
        .dmrfy_exec_addr         (dmrfy_int8_exec_addr),
        .dmrfy_load_done         (dmrfy_load_done[0]),
        .dmrfy_fp_updt_data      (dmrfy_fp_updt_data[0]),
        .dmrfy_fp_updt_en        (dmrfy_fp_updt_en[0]),
        .dmrfy_fp_updt_sel       (dmrfy_fp_updt_sel[0]),
        .dmrfy_fp0_updt_addr     ({4'd0, dmrfy_fp0_updt_addr[0]}),
        .dmrfy_fp0_exec_addr     ({4'd0, dmrfy_fp0_exec_addr[0]}),
        .dmrfy_fp1_updt_addr     ({4'd0, dmrfy_fp1_updt_addr[0]}),
        .dmrfy_fp1_exec_addr     ({4'd0, dmrfy_fp1_exec_addr[0]}),
        .dmrfy_exec_data         (dmrfy_exec_data[0])
    );
    always_ff @(posedge clk) begin
        dmrfy_load_tile_sel[0] <= load_tile_sel;
        dmrfy_mode_sel[0]      <= pc_load_mode_sel;
        dmrfy_int8_load_depth  <= yi_load_depth;
        dmrfy_fp_updt_en[0]    <= fp_updt_en_pipeout;
        dmrfy_fp_updt_sel[0]   <= fp_updt_en_pipeout ? fp_updt_sel_pipeout : pc_fp_load_sel;
        dmrfy_fp0_updt_addr[0] <= fp_updt_en_pipeout ? fp_updt_addr_pipeout : pc_fp_load_vec;
        dmrfy_fp1_updt_addr[0] <= fp_updt_en_pipeout ? fp_updt_addr_pipeout : pc_fp_load_vec;
        dmrfy_fp0_exec_addr[0] <= pc_fp0_exec_vec;
        dmrfy_fp1_exec_addr[0] <= pc_fp1_exec_vec;
    end

    genvar tapu_idx;
    generate
        for (tapu_idx = 1; tapu_idx < TAPU_NUM; tapu_idx = tapu_idx + 1) begin
            dmrf_y #(
                .AXIS_LOAD_DATA_WIDTH(AXIS_LOAD_DATA_WIDTH),
                .SYS_DATA_WIDTH      (Y_SYS_DATA_WIDTH),
                .SYS_PORT_NUM        (Y_SYS_PORT_NUM),
                .BRAM_SIZE           (Y_BRAM_SIZE),
                .BRAM_ADDR_WIDTH     (Y_BRAM_ADDR_WIDTH),
                .BRAM_DATA_WIDTH     (Y_BRAM_DATA_WIDTH),
                .LOAD_ADDR_WIDTH     (Y_LOAD_ADDR_WIDTH),
                .EXEC_ADDR_WIDTH     (Y_EXEC_ADDR_WIDTH)
            ) dmrf_y_fp16_inst (
                .clk                     (clk),
                .rst_n                   (rst_n),
                .s_axis_dmrfy_load_tdata (i_axis_dmrfy_load_tdata[tapu_idx]),
                .s_axis_dmrfy_load_tvalid(i_axis_dmrfy_load_tvalid[tapu_idx]),
                .s_axis_dmrfy_load_tready(i_axis_dmrfy_load_tready[tapu_idx]),
                .s_axis_dmrfy_load_tkeep (i_axis_dmrfy_load_tkeep[tapu_idx]),
                .s_axis_dmrfy_load_tlast (i_axis_dmrfy_load_tlast[tapu_idx]),
                .dmrfy_load_depth        (dmrfy_load_depth[tapu_idx]),
                .dmrfy_mode_sel          (dmrfy_mode_sel[tapu_idx]),
                .dmrfy_load_tile_sel     (dmrfy_load_tile_sel[tapu_idx]),
                .dmrfy_exec_addr         (dmrfy_exec_addr[tapu_idx]),
                .dmrfy_load_done         (dmrfy_load_done[tapu_idx]),
                .dmrfy_fp_updt_data      (dmrfy_fp_updt_data[tapu_idx]),
                .dmrfy_fp_updt_en        (dmrfy_fp_updt_en[tapu_idx]),
                .dmrfy_fp_updt_sel       (dmrfy_fp_updt_sel[tapu_idx]),
                .dmrfy_fp0_updt_addr     (dmrfy_fp0_updt_addr[tapu_idx]),
                .dmrfy_fp0_exec_addr     (dmrfy_fp0_exec_addr[tapu_idx]),
                .dmrfy_fp1_updt_addr     (dmrfy_fp1_updt_addr[tapu_idx]),
                .dmrfy_fp1_exec_addr     (dmrfy_fp1_exec_addr[tapu_idx]),
                .dmrfy_exec_data         (dmrfy_exec_data[tapu_idx])
            );
            always_ff @(posedge clk) begin
                dmrfy_mode_sel[tapu_idx] <= pc_load_mode_sel;
                dmrfy_load_tile_sel[tapu_idx] <= load_tile_sel;  // NOTE: Not used in fp mode
                dmrfy_load_depth[tapu_idx] <= yi_load_depth[4:0];  // NOTE: Not used in fp mode
                dmrfy_fp_updt_en[tapu_idx] <= fp_updt_en_pipeout;
                dmrfy_fp_updt_sel[tapu_idx]   <= fp_updt_en_pipeout ? fp_updt_sel_pipeout : pc_fp_load_sel;
                dmrfy_fp0_updt_addr[tapu_idx] <= fp_updt_en_pipeout ? fp_updt_addr_pipeout : pc_fp_load_vec;
                dmrfy_fp1_updt_addr[tapu_idx] <= fp_updt_en_pipeout ? fp_updt_addr_pipeout : pc_fp_load_vec;
                dmrfy_fp0_exec_addr[tapu_idx] <= pc_fp0_exec_vec;
                dmrfy_fp1_exec_addr[tapu_idx] <= pc_fp1_exec_vec;
            end
        end
    endgenerate
    /****************************************** DMRFY ******************************************/

    /*************************************** DMRF Loading **************************************/
    // @TODO: parameterized
    logic [2:0] dmrfy_dest_sel;
    logic       dmrfy_load_done_t;
    assign dmrfy_load_done_t = dmrfy_load_done[7] | dmrfy_load_done[6] | dmrfy_load_done[5] | dmrfy_load_done[4] | dmrfy_load_done[3] | dmrfy_load_done[2] | dmrfy_load_done[1] | dmrfy_load_done[0];
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            dmrfy_dest_sel <= 3'b000;
        end else if (pc_load_mode_sel & dmrfy_load_done_t) begin
            dmrfy_dest_sel <= dmrfy_dest_sel + 1;
        end
    end

    assign i_axis_dmrfx_load_tdata = s_axis_xi_load_tdata;
    assign i_axis_dmrfx_load_tvalid = s_axis_xi_load_tvalid;
    assign s_axis_xi_load_tready = i_axis_dmrfx_load_tready;
    assign i_axis_dmrfx_load_tkeep = s_axis_xi_load_tkeep;
    assign i_axis_dmrfx_load_tlast = s_axis_xi_load_tlast;

    assign i_axis_dmrfy_load_tdata[0] = s_axis_yi_load_tdata;
    assign i_axis_dmrfy_load_tdata[1] = s_axis_yi_load_tdata;
    assign i_axis_dmrfy_load_tdata[2] = s_axis_yi_load_tdata;
    assign i_axis_dmrfy_load_tdata[3] = s_axis_yi_load_tdata;
    assign i_axis_dmrfy_load_tdata[4] = s_axis_yi_load_tdata;
    assign i_axis_dmrfy_load_tdata[5] = s_axis_yi_load_tdata;
    assign i_axis_dmrfy_load_tdata[6] = s_axis_yi_load_tdata;
    assign i_axis_dmrfy_load_tdata[7] = s_axis_yi_load_tdata;

    assign i_axis_dmrfy_load_tvalid[0] = (dmrfy_dest_sel == 3'b000) ? s_axis_yi_load_tvalid : 1'b0;
    assign i_axis_dmrfy_load_tvalid[1] = (dmrfy_dest_sel == 3'b001) ? s_axis_yi_load_tvalid : 1'b0;
    assign i_axis_dmrfy_load_tvalid[2] = (dmrfy_dest_sel == 3'b010) ? s_axis_yi_load_tvalid : 1'b0;
    assign i_axis_dmrfy_load_tvalid[3] = (dmrfy_dest_sel == 3'b011) ? s_axis_yi_load_tvalid : 1'b0;
    assign i_axis_dmrfy_load_tvalid[4] = (dmrfy_dest_sel == 3'b100) ? s_axis_yi_load_tvalid : 1'b0;
    assign i_axis_dmrfy_load_tvalid[5] = (dmrfy_dest_sel == 3'b101) ? s_axis_yi_load_tvalid : 1'b0;
    assign i_axis_dmrfy_load_tvalid[6] = (dmrfy_dest_sel == 3'b110) ? s_axis_yi_load_tvalid : 1'b0;
    assign i_axis_dmrfy_load_tvalid[7] = (dmrfy_dest_sel == 3'b111) ? s_axis_yi_load_tvalid : 1'b0;

    assign i_axis_dmrfy_load_tkeep[0] = s_axis_yi_load_tkeep;
    assign i_axis_dmrfy_load_tkeep[1] = s_axis_yi_load_tkeep;
    assign i_axis_dmrfy_load_tkeep[2] = s_axis_yi_load_tkeep;
    assign i_axis_dmrfy_load_tkeep[3] = s_axis_yi_load_tkeep;
    assign i_axis_dmrfy_load_tkeep[4] = s_axis_yi_load_tkeep;
    assign i_axis_dmrfy_load_tkeep[5] = s_axis_yi_load_tkeep;
    assign i_axis_dmrfy_load_tkeep[6] = s_axis_yi_load_tkeep;
    assign i_axis_dmrfy_load_tkeep[7] = s_axis_yi_load_tkeep;

    assign i_axis_dmrfy_load_tlast[0] = (dmrfy_dest_sel == 3'b000) ? s_axis_yi_load_tlast : 1'b0;
    assign i_axis_dmrfy_load_tlast[1] = (dmrfy_dest_sel == 3'b001) ? s_axis_yi_load_tlast : 1'b0;
    assign i_axis_dmrfy_load_tlast[2] = (dmrfy_dest_sel == 3'b010) ? s_axis_yi_load_tlast : 1'b0;
    assign i_axis_dmrfy_load_tlast[3] = (dmrfy_dest_sel == 3'b011) ? s_axis_yi_load_tlast : 1'b0;
    assign i_axis_dmrfy_load_tlast[4] = (dmrfy_dest_sel == 3'b100) ? s_axis_yi_load_tlast : 1'b0;
    assign i_axis_dmrfy_load_tlast[5] = (dmrfy_dest_sel == 3'b101) ? s_axis_yi_load_tlast : 1'b0;
    assign i_axis_dmrfy_load_tlast[6] = (dmrfy_dest_sel == 3'b110) ? s_axis_yi_load_tlast : 1'b0;
    assign i_axis_dmrfy_load_tlast[7] = (dmrfy_dest_sel == 3'b111) ? s_axis_yi_load_tlast : 1'b0;

    assign s_axis_yi_load_tready = 1'b1;

    assign pc_loadx_done = dmrfx_load_done;
    assign pc_loady_done = (pc_load_mode_sel & dmrfy_load_done[7]) | ((~pc_load_mode_sel) & dmrfy_load_done[0]);
    /*************************************** DMRF Loading **************************************/

    /****************************************** TAPU *******************************************/
    logic [7:0]                             y_sel_in;
    logic [7:0]                             sys_buf_en_in;
    // from controller
    logic                                   sys_buf_en;
    logic [7:0][     1:0]                   mode_sel_in;
    logic [7:0]                             psu_clr_in;
    logic [7:0][     3:0][  LEFT_WIDTH-1:0] x_in;
    logic [7:0][COLS-1:0][   TOP_WIDTH-1:0] y_in;
    logic [7:0][COLS-1:0][BOTTOM_WIDTH-1:0] z_out;
    logic                                   buf_valid_en;
    logic                                   buf_valid_en_d;

    // always_ff @(posedge clk) begin
    //     exec_en_d <= exec_en;
    // end

    delay_chain #(
        .DW (1),
        .LEN(2)
    ) delay_chain_bufvld_en (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (buf_valid_en),
        .out  (buf_valid_en_d)
    );

    genvar row_idx, col_idx;
    generate
        for (tapu_idx = 0; tapu_idx < TAPU_NUM; tapu_idx = tapu_idx + 1) begin
            tapu #(
                .TAPU_IDX    (tapu_idx),
                .LEFT_WIDTH  (LEFT_WIDTH),
                .RIGHT_WIDTH (RIGHT_WIDTH),
                .TOP_WIDTH   (TOP_WIDTH),
                .BOTTOM_WIDTH(BOTTOM_WIDTH),
                .PRELD_WIDTH (PRELD_WIDTH),
                .COLS        (COLS)
            ) tapu_inst (
                .clk          (clk),
                .rst_n        (rst_n),
                .y_sel_in     (y_sel_in[tapu_idx]),
                .sys_buf_en_in(sys_buf_en_in[tapu_idx]),
                .mode_sel_in  (mode_sel_in[tapu_idx]),
                .psu_clr_in   (psu_clr_in[tapu_idx]),
                .x_in         (x_in[tapu_idx]),
                .y_in         (y_in[tapu_idx]),
                .z_out        (z_out[tapu_idx])
            );
            for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
                always_ff @(posedge clk) begin
                    if (buf_valid_en_d) begin
                        x_in[tapu_idx][row_idx] <= dmrfx_exec_data[((tapu_idx*ROWS+row_idx+1)*X_SYS_DATA_WIDTH-1):((tapu_idx*ROWS+row_idx)*X_SYS_DATA_WIDTH)];
                    end else begin
                        x_in[tapu_idx][row_idx] <= 0;
                    end
                end
                // assign x_in[tapu_idx][row_idx] = dmrfx_exec_data[((tapu_idx*ROWS+row_idx+1)*X_SYS_DATA_WIDTH-1):((tapu_idx*ROWS+row_idx)*X_SYS_DATA_WIDTH)];
            end
            if (tapu_idx == 0) begin
                for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
                    always_ff @(posedge clk) begin
                        if ((~buf_valid_en_d) & (~mode_sel_in[tapu_idx][1])) begin
                            y_in[tapu_idx][col_idx] <= 0;
                        end else begin
                            y_in[tapu_idx][col_idx] <= {
                                16'd0,
                                dmrfy_exec_data[tapu_idx][(col_idx+1)*Y_SYS_DATA_WIDTH-1:col_idx*Y_SYS_DATA_WIDTH]
                            };
                        end
                    end
                    // assign y_in[tapu_idx][col_idx] = {
                    //     16'd0,
                    //     dmrfy_exec_data[tapu_idx][(col_idx+1)*Y_SYS_DATA_WIDTH-1:col_idx*Y_SYS_DATA_WIDTH]
                    // };
                end
            end else begin
                for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
                    assign y_in[tapu_idx][col_idx] = (mode_sel_in[tapu_idx][1] == 1'b1) ? {
                        16'd0,
                        dmrfy_exec_data[tapu_idx][(col_idx+1)*Y_SYS_DATA_WIDTH-1:col_idx*Y_SYS_DATA_WIDTH]
                    } : z_out[tapu_idx-1][col_idx];
                end
            end
            always_ff @(posedge clk) begin
                y_sel_in[tapu_idx]      <= exec_tile_sel;
                sys_buf_en_in[tapu_idx] <= sys_buf_en;
                mode_sel_in[tapu_idx]   <= pc_exec_mode_sel;
            end
            // @TODO: parameterized
            assign dmrfy_fp_updt_data[tapu_idx] = {
                z_out[tapu_idx][15][15:0],
                z_out[tapu_idx][14][15:0],
                z_out[tapu_idx][13][15:0],
                z_out[tapu_idx][12][15:0],
                z_out[tapu_idx][11][15:0],
                z_out[tapu_idx][10][15:0],
                z_out[tapu_idx][9][15:0],
                z_out[tapu_idx][8][15:0],
                z_out[tapu_idx][7][15:0],
                z_out[tapu_idx][6][15:0],
                z_out[tapu_idx][5][15:0],
                z_out[tapu_idx][4][15:0],
                z_out[tapu_idx][3][15:0],
                z_out[tapu_idx][2][15:0],
                z_out[tapu_idx][1][15:0],
                z_out[tapu_idx][0][15:0]
            };
        end
    endgenerate
    /****************************************** TAPU *******************************************/

    /********************************** Execution Controller ***********************************/
    logic                       int8_exec_start;
    logic                       int8_exec_done;
    logic [PSU_DEPTH_WIDTH-1:0] int8_exec_rd_addr;
    // psu_clr_en means the psu reg in each PE should be cleared before execution
    logic                       psu_clr_en;
    // psu_acc_en means the psu reg values will be buffered into the dm fifo
    logic                       psu_acc_en;

    // 00: int8 matmul; 10- fp mul; 11- fp add
    logic [                1:0] exec_mode_sel;
    always_ff @(posedge clk) begin
        exec_mode_sel <= pc_exec_mode_sel;
    end

    exec_ctrl_int8 #(
        .RC_WIDTH       (RC_WIDTH),
        .ROWS           (ROWS),
        .COLS           (COLS),
        .EXTRA_LATENCY  (EXTRA_LATENCY),
        .PSU_DEPTH_WIDTH(PSU_DEPTH_WIDTH)
    ) exec_ctrl_int8_inst (
        .clk              (clk),
        .rst_n            (rst_n),
        .exec_start       (int8_exec_start),
        .exec_done        (int8_exec_done),
        .buf_valid_en     (buf_valid_en),
        .psu_acc_en       (psu_acc_en),
        .psu_depth        (psu_fpv_depth),
        .int8_exec_rd_addr(int8_exec_rd_addr),
        .sys_buf_en       (sys_buf_en)
    );

    generate
        for (tapu_idx = 0; tapu_idx < TAPU_NUM; tapu_idx = tapu_idx + 1) begin
            always_ff @(posedge clk) begin
                dmrfy_exec_addr[tapu_idx] <= int8_exec_rd_addr[4:0];
                psu_clr_in[tapu_idx]      <= psu_clr_en;
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        dmrfx_exec_addr      <= int8_exec_rd_addr;
        dmrfy_int8_exec_addr <= int8_exec_rd_addr;
    end

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            int8_exec_start <= 1'b0;
        end else begin
            int8_exec_start <= pc_exec_start & ~pc_exec_mode_sel[1];
        end
    end

    assign psu_clr_en = pc_exec_start & (~pc_exec_mode_sel[1]) & psu_clr;
    assign psu_acc_en = (~pc_exec_mode_sel[1]) & psu_acc;
    assign pc_exec_done = exec_mode_sel[1] ? (fp_updt_en_pipeout | fp_wmem_en_pipeout) : int8_exec_done;

    // delay 3 cycles of sys_buf_en to get the dm fifo write enable in int8 mode
    logic dmb_buf_en_int8;
    delay_chain #(
        .DW (1),
        .LEN(3)
    ) delay_dmb_buf_en_int8 (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (sys_buf_en),
        .out  (dmb_buf_en_int8)
    );
    /********************************** Execution Controller ***********************************/

    /*********************************** Output Controller *************************************/
    logic       zout_start;
    logic       zout_done;
    logic       tapu_z_out_en;
    logic       mode_sel_zout;
    logic [2:0] tapu_store_idx;

    zout_ctrl zout_ctrl_inst (
        .clk             (clk),
        .rst_n           (rst_n),
        .zout_start      (zout_start),
        .zout_done       (zout_done),
        .store_depth     (store_depth),
        .store_tapu_depth(store_tapu_depth),
        .psu_store_en    (tapu_z_out_en),
        .tapu_store_idx  (tapu_store_idx)
    );

    always_ff @(posedge clk) begin
        mode_sel_zout <= pc_store_mode_sel[1];
    end

    always_ff @(posedge clk) begin
        pc_store_done <= zout_done;
    end

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            zout_start <= 1'b0;
        end else begin
            zout_start <= pc_store_start;
        end
    end
    /*********************************** Output Controller *************************************/

    /******************************** Dual Mode Buffer (Small) *********************************/
    logic [TAPU_NUM-1:0]                          dmb_fifo_empty;
    logic [TAPU_NUM-1:0]                          dmb_fifo_full;
    logic [TAPU_NUM-1:0]                          dmb_fifo_rd_en;
    logic [TAPU_NUM-1:0]                          dmb_fifo_wr_en;
    logic [TAPU_NUM-1:0][DMB_FIFO_DATA_WIDTH-1:0] dmb_fifo_data_in;
    logic [TAPU_NUM-1:0][DMB_FIFO_DATA_WIDTH-1:0] dmb_fifo_data_out;

    generate
        for (tapu_idx = 0; tapu_idx < TAPU_NUM; tapu_idx = tapu_idx + 1) begin
            fifo_common #(
                .WR_DATA_WIDTH   (DMB_FIFO_DATA_WIDTH),
                .FIFO_DEPTH      (DMB_FIFO_DEPTH),
                .FIFO_MEMORY_TYPE("auto")
            ) dmb_fifo_inst (
                .clk  (clk),
                .rst_n(rst_n),
                .rd_en(dmb_fifo_rd_en[tapu_idx]),
                .wr_en(dmb_fifo_wr_en[tapu_idx]),
                .empty(dmb_fifo_empty[tapu_idx]),
                .full (dmb_fifo_full[tapu_idx]),
                .din  (dmb_fifo_data_in[tapu_idx]),
                .dout (dmb_fifo_data_out[tapu_idx])
            );
            assign dmb_fifo_data_in[tapu_idx] = {
                z_out[tapu_idx][15][31:0],
                z_out[tapu_idx][14][31:0],
                z_out[tapu_idx][13][31:0],
                z_out[tapu_idx][12][31:0],
                z_out[tapu_idx][11][31:0],
                z_out[tapu_idx][10][31:0],
                z_out[tapu_idx][9][31:0],
                z_out[tapu_idx][8][31:0],
                z_out[tapu_idx][7][31:0],
                z_out[tapu_idx][6][31:0],
                z_out[tapu_idx][5][31:0],
                z_out[tapu_idx][4][31:0],
                z_out[tapu_idx][3][31:0],
                z_out[tapu_idx][2][31:0],
                z_out[tapu_idx][1][31:0],
                z_out[tapu_idx][0][31:0]
            };
            assign dmb_fifo_wr_en[tapu_idx] = fp_wmem_en_pipeout | dmb_buf_en_int8;
            assign dmb_fifo_rd_en[tapu_idx] = tapu_z_out_en & (tapu_store_idx == tapu_idx);
        end
    endgenerate

    /******************************** Dual Mode Buffer (Small) *********************************/

    /*************************************** Quantization ***************************************/
    logic        [QUANT_SCALE_EXP_WIDTH-1:0] sf_exp_r;
    logic signed [QUANT_SCALE_MAN_WIDTH-1:0] sf_man_r;
    // quant mode. 00: int8 -> int8, 01: int8 -> fp16, 10: fp16 -> int8, 11: fp16 -> fp16
    logic        [                      1:0] quant_mode;
    logic        [  DMB_FIFO_DATA_WIDTH-1:0] quant_data_in;
    logic                                    quant_data_in_valid;
    logic                                    quant_data_in_last;
    logic        [ AXIS_S2MM_DATA_WIDTH-1:0] quant_s2mm_out;
    logic                                    quant_s2mm_out_valid;
    logic                                    quant_s2mm_out_last;
    logic        [                      2:0] tapu_store_idx_d;
    logic        [                      7:0] quant_int8_fp16_depth;

    always_ff @(posedge clk) begin
        quant_int8_fp16_depth <= {store_depth, 1'b0};
    end

    always_ff @(posedge clk) begin
        sf_exp_r   <= sf_exp;
        sf_man_r   <= sf_man;
        quant_mode <= pc_store_mode_sel;
    end

    // for int8, we should delay 2 cycles to optimize the timing
    delay_chain #(
        .DW (1),
        .LEN(2)
    ) delay_int8_store_valid (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (tapu_z_out_en),
        .out  (quant_data_in_valid)
    );

    delay_chain #(
        .DW (1),
        .LEN(2)
    ) delay_int8_store_last (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (zout_done),
        .out  (quant_data_in_last)
    );

    // MUX the data
    always_ff @(posedge clk) begin
        tapu_store_idx_d <= tapu_store_idx;
    end

    always_ff @(posedge clk) begin
        quant_data_in <= dmb_fifo_data_out[tapu_store_idx_d];
    end

    dm_quant #(
        .QUANT_SCALE_EXP_WIDTH(QUANT_SCALE_EXP_WIDTH),
        .QUANT_SCALE_MAN_WIDTH(QUANT_SCALE_MAN_WIDTH),
        .FPM_DATA_WIDTH       (FPM_DATA_WIDTH),
        .DMB_FIFO_DATA_WIDTH  (DMB_FIFO_DATA_WIDTH),
        .AXIS_S2MM_DATA_WIDTH (AXIS_S2MM_DATA_WIDTH),
        .COLS                 (COLS),
        .PSU_DATA_WIDTH       (PSU_DATA_WIDTH)
    ) dm_quant_inst (
        .clk                  (clk),
        .rst_n                (rst_n),
        .sf_exp_r             (sf_exp_r),
        .sf_man_r             (sf_man_r),
        .quant_mode           (quant_mode),
        .quant_int8_fp16_depth(quant_int8_fp16_depth),
        .quant_data_in        (quant_data_in),
        .quant_data_in_valid  (quant_data_in_valid),
        .quant_data_in_last   (quant_data_in_last),
        .quant_s2mm_out       (quant_s2mm_out),
        .quant_s2mm_out_valid (quant_s2mm_out_valid),
        .quant_s2mm_out_last  (quant_s2mm_out_last)
    );
    /*************************************** Quantization ***************************************/

    /****************************************** S2MM *******************************************/
    // s2mm fifo to avoid stalling
    logic                              i_axis_s2mm_fifo_tvalid;
    logic                              i_axis_s2mm_fifo_tready;
    logic [  AXIS_S2MM_DATA_WIDTH-1:0] i_axis_s2mm_fifo_tdata;
    logic [AXIS_S2MM_DATA_WIDTH/8-1:0] i_axis_s2mm_fifo_tkeep;
    logic                              i_axis_s2mm_fifo_tlast;

    fifo_axis #(
        .FIFO_AXIS_DEPTH      (64),
        .FIFO_AXIS_TDATA_WIDTH(AXIS_S2MM_DATA_WIDTH)
    ) fifo_axis_s2mm_inst (
        //common signal
        .clk          (clk),
        .rst_n        (rst_n),
        // s_axis
        .s_axis_tready(i_axis_s2mm_fifo_tready),
        .s_axis_tvalid(i_axis_s2mm_fifo_tvalid),
        .s_axis_tdata (i_axis_s2mm_fifo_tdata),
        .s_axis_tkeep (i_axis_s2mm_fifo_tkeep),
        .s_axis_tlast (i_axis_s2mm_fifo_tlast),
        // m_axis
        .m_axis_tready(m_axis_s2mm_tready),
        .m_axis_tvalid(m_axis_s2mm_tvalid),
        .m_axis_tdata (m_axis_s2mm_tdata),
        .m_axis_tkeep (m_axis_s2mm_tkeep),
        .m_axis_tlast (m_axis_s2mm_tlast)
    );

    assign i_axis_s2mm_fifo_tvalid = quant_s2mm_out_valid;
    assign i_axis_s2mm_fifo_tdata  = quant_s2mm_out;
    assign i_axis_s2mm_fifo_tlast  = quant_s2mm_out_last;
    assign i_axis_s2mm_fifo_tkeep  = 32'hffffffff;

    /****************************************** S2MM *******************************************/

endmodule
