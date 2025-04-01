`timescale 1ns / 1ps

// the matrix transpose only works for int8
module transpose_int #(
    parameter MT_PORT_NUM   = 32,
    parameter MT_DATA_WIDTH = 8
) (
    input  logic                                 clk,
    input  logic                                 rst_n,
    input  logic                                 mt_en,
    input  logic [MT_PORT_NUM*MT_DATA_WIDTH-1:0] mt_data_in,
    input  logic                                 mt_data_in_valid,
    input  logic                                 mt_data_in_last,
    output logic [MT_PORT_NUM*MT_DATA_WIDTH-1:0] mt_data_out,
    output logic                                 mt_data_out_valid,
    output logic                                 mt_data_out_last
);

    // regs for transpose
    logic [MT_PORT_NUM-1:0][MT_PORT_NUM-1:0][MT_DATA_WIDTH-1:0] mt_regs;

    // connect as systolic array
    genvar row_idx, col_idx;
    generate
        for (row_idx = 0; row_idx < MT_PORT_NUM; row_idx = row_idx + 1) begin
            for (col_idx = 0; col_idx < MT_PORT_NUM; col_idx = col_idx + 1) begin
                if (row_idx == 0 && col_idx == 0) begin
                    always_ff @(posedge clk) begin
                        if (~rst_n) begin
                            mt_regs[row_idx][col_idx] <= 0;
                        end else if (mt_data_in_valid) begin
                            mt_regs[row_idx][col_idx] <= mt_data_in[(MT_PORT_NUM-col_idx)*MT_DATA_WIDTH-1:(MT_PORT_NUM-col_idx-1)*MT_DATA_WIDTH];
                        end
                    end
                end else if (row_idx == 0) begin
                    always_ff @(posedge clk) begin
                        if (~rst_n) begin
                            mt_regs[row_idx][col_idx] <= 0;
                        end else if (mt_data_in_valid) begin
                            mt_regs[row_idx][col_idx] <= mt_data_in[(MT_PORT_NUM-col_idx)*MT_DATA_WIDTH-1:(MT_PORT_NUM-col_idx-1)*MT_DATA_WIDTH];
                        end else if (mt_data_out_valid) begin
                            mt_regs[row_idx][col_idx] <= mt_regs[row_idx][col_idx-1];
                        end
                    end
                end else if (col_idx == 0) begin
                    always_ff @(posedge clk) begin
                        if (~rst_n) begin
                            mt_regs[row_idx][col_idx] <= 0;
                        end else if (mt_data_in_valid) begin
                            mt_regs[row_idx][col_idx] <= mt_regs[row_idx-1][col_idx];
                        end
                    end
                end else begin
                    always_ff @(posedge clk) begin
                        if (~rst_n) begin
                            mt_regs[row_idx][col_idx] <= 0;
                        end else if (mt_data_in_valid) begin
                            mt_regs[row_idx][col_idx] <= mt_regs[row_idx-1][col_idx];
                        end else if (mt_data_out_valid) begin
                            mt_regs[row_idx][col_idx] <= mt_regs[row_idx][col_idx-1];
                        end
                    end
                end
            end
        end
    endgenerate

    generate
        for (row_idx = 0; row_idx < MT_PORT_NUM; row_idx = row_idx + 1) begin
            assign mt_data_out[row_idx*MT_DATA_WIDTH+MT_DATA_WIDTH-1:row_idx*MT_DATA_WIDTH] = mt_regs[row_idx][MT_PORT_NUM-1];
        end
    endgenerate

    // counter for input, 5-bit -> 32
    // TODO: parameterize
    logic [4:0] cnt_mt_32;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            cnt_mt_32 <= 0;
        end else if (mt_data_in_valid) begin
            cnt_mt_32 <= cnt_mt_32 + 1;
        end
    end

    logic [4:0] cnt_mt_out_32;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            mt_data_out_valid <= 1'b0;
        end else if (mt_en & mt_data_in_valid & (cnt_mt_32 == 5'd31)) begin
            mt_data_out_valid <= 1'b1;
        end else if (mt_en & mt_data_out_valid & (cnt_mt_out_32 == 5'd31)) begin
            mt_data_out_valid <= 1'b0;
        end
    end
    assign mt_data_out_last = mt_en & mt_data_out_valid & (cnt_mt_out_32 == 5'd31);

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            cnt_mt_out_32 <= 0;
        end else if (mt_data_out_valid) begin
            cnt_mt_out_32 <= cnt_mt_out_32 + 1;
        end
    end

endmodule
