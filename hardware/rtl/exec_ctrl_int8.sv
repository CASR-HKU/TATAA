`timescale 1ns / 1ps

// CONFIGURATION:
// - Maximum depth for one stream: 512
module exec_ctrl_int8 #(
    parameter RC_WIDTH        = 2,
    parameter ROWS            = 4,
    parameter COLS            = 16,
    parameter EXTRA_LATENCY   = 51,
    parameter PSU_DEPTH_WIDTH = 9
) (
    // common signals
    input  logic                       clk,
    input  logic                       rst_n,
    // control signals from top controller in this core, and output
    input  logic                       exec_start,
    output logic                       exec_done,
    output logic                       buf_valid_en,
    input  logic                       psu_acc_en,
    input  logic [PSU_DEPTH_WIDTH-1:0] psu_depth,
    output logic [PSU_DEPTH_WIDTH-1:0] int8_exec_rd_addr,
    output logic                       sys_buf_en
);

    logic exec_en;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            exec_en <= 1'b0;
        end else if (full_cnt_tile_flow) begin
            exec_en <= 1'b0;
        end else if (exec_start) begin
            exec_en <= 1'b1;
        end
    end

    // counter for one tile
    logic [PSU_DEPTH_WIDTH:0] cnt_tile_flow;
    logic                     full_cnt_tile_flow;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            cnt_tile_flow <= 0;
        end else if (exec_en) begin
            if (cnt_tile_flow == (psu_depth + EXTRA_LATENCY)) begin
                cnt_tile_flow <= 0;
            end else begin
                cnt_tile_flow <= cnt_tile_flow + 1;
            end
        end
    end
    assign full_cnt_tile_flow = exec_en & (cnt_tile_flow == (psu_depth + EXTRA_LATENCY));

    // counter for sys_buf_en
    logic [1:0] cnt_sys_buf_en;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            cnt_sys_buf_en <= 0;
        end else if (sys_buf_en) begin
            cnt_sys_buf_en <= cnt_sys_buf_en + 1;
        end
    end

    assign exec_done = psu_acc_en ? (sys_buf_en & (cnt_sys_buf_en == 2'b11)) : full_cnt_tile_flow;
    assign buf_valid_en = exec_en & (cnt_tile_flow[PSU_DEPTH_WIDTH-1:0] <= psu_depth);

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            int8_exec_rd_addr <= 0;
        end else if (buf_valid_en) begin
            int8_exec_rd_addr <= int8_exec_rd_addr + 1;
        end
    end

    // in the final 4 cycles, activate sys_buf_en:
    // assign sys_buf_en = exec_en & psu_acc_en & (cnt_tile_flow > (psu_depth + EXTRA_LATENCY - 4));
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            sys_buf_en <= 1'b0;
        end else if (full_cnt_tile_flow & psu_acc_en) begin
            sys_buf_en <= 1'b1;
        end else if (sys_buf_en & (cnt_sys_buf_en == 2'b11)) begin
            sys_buf_en <= 1'b0;
        end
    end

endmodule
