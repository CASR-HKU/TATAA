`timescale 1ns / 1ps

// @TODO: discuss the overflow issue in bfp8
(* keep_hierarchy = "yes" *)
module tapu #(
    parameter TAPU_IDX     = 0,
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
    // input  logic [     2:0]                   fp_op_sel,
    // data input
    input  logic [     3:0][  LEFT_WIDTH-1:0] x_in,
    input  logic [COLS-1:0][   TOP_WIDTH-1:0] y_in,
    // data output, including psu in bfp8 mode and bfloat16 in fp mode
    output logic [COLS-1:0][BOTTOM_WIDTH-1:0] z_out
);

    /****************************************** Delay Chain ******************************************/
    logic [     3:0][LEFT_WIDTH-1:0] sys_x_in;
    logic [COLS-1:0][ TOP_WIDTH-1:0] sys_y_in_matmul;
    logic [COLS-1:0][ TOP_WIDTH-1:0] sys_y_in;

    genvar idx_row, idx_col;
    generate
        for (idx_row = 0; idx_row < 4; idx_row = idx_row + 1) begin
            delay_chain #(
                .DW (LEFT_WIDTH),
                .LEN(TAPU_IDX * 4 + idx_row)
            ) delay_chain_x_in (
                .clk  (clk),
                .rst_n(rst_n),
                .en   (1'b1),
                .in   (x_in[idx_row]),
                .out  (sys_x_in[idx_row])
            );
        end
    endgenerate

    // if the TAPU_IDX == 0, the y_in needs to be delayed in matmul
    generate
        if (TAPU_IDX == 0) begin
            for (idx_col = 0; idx_col < COLS; idx_col = idx_col + 1) begin
                delay_chain #(
                    .DW (TOP_WIDTH),
                    .LEN(idx_col)
                ) delay_chain_y_in (
                    .clk  (clk),
                    .rst_n(rst_n),
                    .en   (1'b1),
                    .in   (y_in[idx_col]),
                    .out  (sys_y_in_matmul[idx_col])
                );
                always_ff @( posedge clk ) begin
                    sys_y_in[idx_col] <= (mode_sel_in == 2'b00) ? sys_y_in_matmul[idx_col] : y_in[idx_col];
                end
                // assign sys_y_in[idx_col] = (mode_sel_in == 2'b00) ? sys_y_in_matmul[idx_col] : y_in[idx_col];
            end
        end else begin
            for (idx_col = 0; idx_col < COLS; idx_col = idx_col + 1) begin
                always_ff @( posedge clk ) begin
                    sys_y_in[idx_col] <= y_in[idx_col];
                end
                // assign sys_y_in[idx_col] = y_in[idx_col];
            end
        end
    endgenerate
    /****************************************** Delay Chain ******************************************/

    /******************************************* PE Array ********************************************/
    pe_sys #(
        .LEFT_WIDTH  (LEFT_WIDTH),
        .RIGHT_WIDTH (RIGHT_WIDTH),
        .TOP_WIDTH   (TOP_WIDTH),
        .BOTTOM_WIDTH(BOTTOM_WIDTH),
        .PRELD_WIDTH (PRELD_WIDTH),
        .COLS        (COLS)
    ) pe_sys (
        .clk          (clk),
        .rst_n        (rst_n),
        .y_sel_in     (y_sel_in),
        .sys_buf_en_in(sys_buf_en_in),
        .mode_sel_in  (mode_sel_in),
        .psu_clr_in   (psu_clr_in),
        // .fp_op_sel    (fp_op_sel),
        .left_in      (sys_x_in),
        .top_in       (sys_y_in),
        .bottom_out   (z_out)
    );
    /******************************************* PE Array ********************************************/

endmodule
