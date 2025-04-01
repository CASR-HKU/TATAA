`timescale 1ns / 1ps

// PE array
(* keep_hierarchy = "yes" *)
module pe_sys #(
    parameter LEFT_WIDTH   = 8,
    parameter RIGHT_WIDTH  = 8,
    parameter TOP_WIDTH    = 48,
    parameter BOTTOM_WIDTH = 48,
    parameter PRELD_WIDTH  = 16,
    parameter COLS         = 16
) (
    // common signals
    input  logic                              clk,
    input  logic                              rst_n,
    // y_sel_in is used to indicate Y tile.
    input  logic                              y_sel_in,
    // store_en is used output
    input  logic                              sys_buf_en_in,
    // 00: matrix multiplication, 10: fp mul, 11: fp add
    input  logic [     1:0]                   mode_sel_in,
    // psu clear
    input  logic                              psu_clr_in,
    // systolic array signals
    input  logic [     3:0][  LEFT_WIDTH-1:0] left_in,
    input  logic [COLS-1:0][   TOP_WIDTH-1:0] top_in,
    output logic [COLS-1:0][BOTTOM_WIDTH-1:0] bottom_out
);

    // connection signals
    logic [COLS-1:0][BOTTOM_WIDTH-1:0] col_connection_01;
    logic [COLS-1:0][BOTTOM_WIDTH-1:0] col_connection_12;
    logic [COLS-1:0][BOTTOM_WIDTH-1:0] col_connection_23;
    logic [COLS+1:0][  LEFT_WIDTH-1:0] row_connection_0;
    logic [COLS+1:0][  LEFT_WIDTH-1:0] row_connection_1;
    logic [COLS+1:0][  LEFT_WIDTH-1:0] row_connection_2;
    logic [COLS+1:0][  LEFT_WIDTH-1:0] row_connection_3;

    assign row_connection_0[0] = left_in[0];
    assign row_connection_1[0] = left_in[1];
    assign row_connection_2[0] = left_in[2];
    assign row_connection_3[0] = left_in[3];

    // optimize the fanout
    logic [COLS-1:0]      y_sel_in_r;
    logic [COLS-1:0]      sys_buf_en_in_r;
    logic [COLS-1:0][1:0] mode_sel_in_r;
    logic [COLS-1:0]      psu_clr_in_r;

    // pe array
    genvar col_idx;
    generate
        for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            always_ff @(posedge clk) begin
                if (~rst_n) begin
                    y_sel_in_r[col_idx]      <= 1'b0;
                    sys_buf_en_in_r[col_idx] <= 1'b0;
                    mode_sel_in_r[col_idx]   <= 2'b00;
                    psu_clr_in_r[col_idx]    <= 1'b0;
                end else begin
                    y_sel_in_r[col_idx]      <= y_sel_in;
                    sys_buf_en_in_r[col_idx] <= sys_buf_en_in;
                    mode_sel_in_r[col_idx]   <= mode_sel_in;
                    psu_clr_in_r[col_idx]    <= psu_clr_in;
                end
            end
            pe_stg_0 #(
                .LEFT_WIDTH  (LEFT_WIDTH  ),
                .RIGHT_WIDTH (RIGHT_WIDTH ),
                .TOP_WIDTH   (TOP_WIDTH   ),
                .BOTTOM_WIDTH(BOTTOM_WIDTH),
                .PRELD_WIDTH (PRELD_WIDTH )
            ) pe_stg_0_inst (
                .clk          (clk),
                .rst_n        (rst_n),
                .y_sel_in     (y_sel_in_r[col_idx]),
                .sys_buf_en_in(sys_buf_en_in_r[col_idx]),
                .mode_sel_in  (mode_sel_in_r[col_idx]),
                .psu_clr_in   (psu_clr_in_r[col_idx]),
                .left_in      (row_connection_0[col_idx]),
                .right_out    (row_connection_0[col_idx+1]),
                .top_in       (top_in[col_idx]),
                .bottom_out   (col_connection_01[col_idx])
            );
            pe_stg_1 #(
                .LEFT_WIDTH  (LEFT_WIDTH  ),
                .RIGHT_WIDTH (RIGHT_WIDTH ),
                .TOP_WIDTH   (TOP_WIDTH   ),
                .BOTTOM_WIDTH(BOTTOM_WIDTH),
                .PRELD_WIDTH (PRELD_WIDTH )
            ) pe_stg_1_inst (
                .clk          (clk),
                .rst_n        (rst_n),
                .y_sel_in     (y_sel_in_r[col_idx]),
                .sys_buf_en_in(sys_buf_en_in_r[col_idx]),
                .mode_sel_in  (mode_sel_in_r[col_idx]),
                .psu_clr_in   (psu_clr_in_r[col_idx]),
                .left_in      (row_connection_1[col_idx]),
                .right_out    (row_connection_1[col_idx+1]),
                .top_in       (col_connection_01[col_idx]),
                .bottom_out   (col_connection_12[col_idx])
            );
            pe_stg_2 #(
                .LEFT_WIDTH  (LEFT_WIDTH  ),
                .RIGHT_WIDTH (RIGHT_WIDTH ),
                .TOP_WIDTH   (TOP_WIDTH   ),
                .BOTTOM_WIDTH(BOTTOM_WIDTH),
                .PRELD_WIDTH (PRELD_WIDTH )
            ) pe_stg_2_inst (
                .clk          (clk),
                .rst_n        (rst_n),
                .y_sel_in     (y_sel_in_r[col_idx]),
                .sys_buf_en_in(sys_buf_en_in_r[col_idx]),
                .mode_sel_in  (mode_sel_in_r[col_idx]),
                .psu_clr_in   (psu_clr_in_r[col_idx]),
                .left_in      (row_connection_2[col_idx]),
                .right_out    (row_connection_2[col_idx+1]),
                .top_in       (col_connection_12[col_idx]),
                .bottom_out   (col_connection_23[col_idx])
            );
            pe_stg_3 #(
                .LEFT_WIDTH  (LEFT_WIDTH  ),
                .RIGHT_WIDTH (RIGHT_WIDTH ),
                .TOP_WIDTH   (TOP_WIDTH   ),
                .BOTTOM_WIDTH(BOTTOM_WIDTH),
                .PRELD_WIDTH (PRELD_WIDTH )
            ) pe_stg_3_inst (
                .clk          (clk),
                .rst_n        (rst_n),
                .y_sel_in     (y_sel_in_r[col_idx]),
                .sys_buf_en_in(sys_buf_en_in_r[col_idx]),
                .mode_sel_in  (mode_sel_in_r[col_idx]),
                .psu_clr_in   (psu_clr_in_r[col_idx]),
                .left_in      (row_connection_3[col_idx]),
                .right_out    (row_connection_3[col_idx+1]),
                .top_in       (col_connection_23[col_idx]),
                .bottom_out   (bottom_out[col_idx])
            );
        end
    endgenerate

endmodule
