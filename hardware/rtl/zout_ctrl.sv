`timescale 1ns / 1ps

module zout_ctrl (
    // common signals
    input  logic       clk,
    input  logic       rst_n,
    // control signals from top controller in this core, and output
    input  logic       zout_start,
    output logic       zout_done,
    input  logic [7:0] store_depth,
    input  logic [4:0] store_tapu_depth,
    output logic       psu_store_en,
    output logic [2:0] tapu_store_idx
);

    // state
    logic zout_state;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            zout_state <= 1'b0;
        end else if (zout_done) begin
            zout_state <= 1'b0;
        end else if (zout_start) begin
            zout_state <= 1'b1;
        end
    end

    // control 
    logic [7:0] cnt_zout;
    logic       full_cnt_zout;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            cnt_zout <= 0;
        end else if (zout_state) begin
            if (cnt_zout == store_depth) begin
                cnt_zout <= 0;
            end else begin
                cnt_zout <= cnt_zout + 1;
            end
        end
    end
    assign full_cnt_zout = zout_state & (cnt_zout == store_depth);

    assign zout_done     = full_cnt_zout;
    assign psu_store_en  = zout_state;

    logic [4:0] cnt_fpm_each_tapu;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            cnt_fpm_each_tapu <= 0;
        end else if (zout_state) begin
            if (cnt_fpm_each_tapu == store_tapu_depth) begin
                cnt_fpm_each_tapu <= 0;
            end else begin
                cnt_fpm_each_tapu <= cnt_fpm_each_tapu + 1;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            tapu_store_idx <= 0;
        end else if (zout_state & (cnt_fpm_each_tapu == store_tapu_depth)) begin
            tapu_store_idx <= tapu_store_idx + 1;
        end
    end

endmodule
