`timescale 1ns / 1ps

// Dual mode quantization
module dm_quant #(
    parameter QUANT_SCALE_EXP_WIDTH = 5,
    parameter QUANT_SCALE_MAN_WIDTH = 16,
    parameter FPM_DATA_WIDTH        = 16,
    parameter DMB_FIFO_DATA_WIDTH   = 512,
    parameter AXIS_S2MM_DATA_WIDTH  = 256,
    parameter COLS                  = 16,
    parameter PSU_DATA_WIDTH        = 32
) (
    // common signals
    input  logic                                                        clk,
    input  logic                                                        rst_n,
    // scaling factors
    input  logic        [QUANT_SCALE_EXP_WIDTH-1:0]                     sf_exp_r,
    input  logic        [QUANT_SCALE_MAN_WIDTH-1:0]                     sf_man_r,
    // quant mode. 00: int8 -> int8, 01: int8 -> fp16, 10: fp16 -> int8, 11: fp16 -> fp16
    input  logic        [                      1:0]                     quant_mode,
    // TODO: parameterize. quant_int8_fp16_depth = 2 * store_depth
    input  logic        [                      7:0]                     quant_int8_fp16_depth,
    input  logic        [  DMB_FIFO_DATA_WIDTH-1:0]                     quant_data_in,
    input  logic                                                        quant_data_in_valid,
    input  logic                                                        quant_data_in_last,
    // output data and valid signals. Note that the valid signals are generated inside
    output logic        [ AXIS_S2MM_DATA_WIDTH-1:0]                     quant_s2mm_out,
    output logic                                                        quant_s2mm_out_valid,
    output logic                                                        quant_s2mm_out_last
);

    /************************************** Shared Part **************************************/
    logic signed [COLS-1:0][8:0] quant_data_in_man;
    logic        [COLS-1:0][8:0] quant_data_in_man_sm;
    genvar col_idx;
    generate
        for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            assign quant_data_in_man_sm[col_idx] = {
                quant_data_in[FPM_DATA_WIDTH-1+FPM_DATA_WIDTH*col_idx],
                1'b1,
                quant_data_in[FPM_DATA_WIDTH*col_idx+6:FPM_DATA_WIDTH*col_idx]
            };
            sm_twos_convert sm_twos_fp16_quant (
                .signed_mag_in      (quant_data_in_man_sm[col_idx]),
                .twos_complement_out(quant_data_in_man[col_idx])
            );
        end
    endgenerate

    // in int8 -> int8, int8 -> fp16, fp16 -> int8 modes, the sf_man should always be multiplied by input
    (* use_dsp = "yes" *)logic signed [COLS-1:0][47:0] mul_sf_man_0;
    (* use_dsp = "yes" *)logic signed [COLS-1:0][47:0] mul_sf_man_1;
    // in int8 -> int8, fp16 -> int8 modes, the results should always be shifted
    logic signed [COLS-1:0][47:0] shift_sf_0;
    logic signed [COLS-1:0][47:0] shift_sf_1;
    generate
        for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            always_ff @(posedge clk) begin
                if (quant_mode[1] == 1'b0) begin
                    mul_sf_man_0[col_idx] <= quant_data_in[col_idx * 32 + 15:col_idx * 32] * sf_man_r;
                    mul_sf_man_1[col_idx] <= quant_data_in[col_idx * 32 + 16 + 15:col_idx * 32 + 16] * sf_man_r;
                end else begin
                    mul_sf_man_0[col_idx] <= quant_data_in_man[col_idx] * sf_man_r;
                end
            end
            bs_rsf bs_rsf_sf_0 (
                .a(mul_sf_man_0[col_idx]),
                .b(sf_exp_r),
                .c(shift_sf_0[col_idx])
            );
            bs_rsf bs_rsf_sf_1 (
                .a(mul_sf_man_1[col_idx]),
                .b(sf_exp_r),
                .c(shift_sf_1[col_idx])
            );
        end
    endgenerate
    /************************************** Shared Part **************************************/

    /************************************** int8 -> int8 **************************************/
    logic [                COLS-1:0][7:0] int8_int8_quant_0;
    logic [                COLS-1:0][7:0] int8_int8_quant_1;
    logic                                 int8_int8_valid;
    logic [AXIS_S2MM_DATA_WIDTH-1:0]      int8_int8_data_out;
    logic                                 int8_int8_last;

    // for int8 -> int8 mode, the quantization is just truncate
    generate
        for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            always_ff @(posedge clk) begin
                int8_int8_quant_0[col_idx] <= shift_sf_0[col_idx][7:0];
                int8_int8_quant_1[col_idx] <= shift_sf_1[col_idx][7:0];
            end
        end
    endgenerate

    // valid signal is 2-cycle delayed
    delay_chain #(
        .DW (1),
        .LEN(2)
    ) delay_int8_int8_valid (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (quant_data_in_valid),
        .out  (int8_int8_valid)
    );

    delay_chain #(
        .DW (1),
        .LEN(2)
    ) delay_int8_int8_last (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (quant_data_in_last),
        .out  (int8_int8_last)
    );

    // combine the results
    assign int8_int8_data_out = {
        int8_int8_quant_1[15],
        int8_int8_quant_0[15],
        int8_int8_quant_1[14],
        int8_int8_quant_0[14],
        int8_int8_quant_1[13],
        int8_int8_quant_0[13],
        int8_int8_quant_1[12],
        int8_int8_quant_0[12],
        int8_int8_quant_1[11],
        int8_int8_quant_0[11],
        int8_int8_quant_1[10],
        int8_int8_quant_0[10],
        int8_int8_quant_1[9],
        int8_int8_quant_0[9],
        int8_int8_quant_1[8],
        int8_int8_quant_0[8],
        int8_int8_quant_1[7],
        int8_int8_quant_0[7],
        int8_int8_quant_1[6],
        int8_int8_quant_0[6],
        int8_int8_quant_1[5],
        int8_int8_quant_0[5],
        int8_int8_quant_1[4],
        int8_int8_quant_0[4],
        int8_int8_quant_1[3],
        int8_int8_quant_0[3],
        int8_int8_quant_1[2],
        int8_int8_quant_0[2],
        int8_int8_quant_1[1],
        int8_int8_quant_0[1],
        int8_int8_quant_1[0],
        int8_int8_quant_0[0]
    };
    /************************************** int8 -> int8 **************************************/

    /************************************** int8 -> fp16 **************************************/
    // in this mode, we don't need to shift the mul result,
    // just regard the mul result as the mantissa of the fp16 number, and normalize it
    // the 48-bit multiplication results are fixed-point format with 16-bit mantissa, 1-bit sign, and 31-bit integer part

    // first, convert int48 data to sign-ed magnitude format if the sign bit is 1
    logic signed [COLS-1:0][47:0] sm_int48_0;
    logic signed [COLS-1:0][47:0] sm_int48_1;
    logic signed [COLS-1:0][47:0] sm_mul_sf_man_0;
    logic signed [COLS-1:0][47:0] sm_mul_sf_man_1;
    logic signed [COLS-1:0]       sm_sign_sf_man_0;
    logic signed [COLS-1:0]       sm_sign_sf_man_1;
    generate
        for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            twos_sm_convert #(
                .DW(48)
            ) twos_sm_convert_int48_0 (
                .twos_complement_in(mul_sf_man_0[col_idx]),
                .signed_mag_out    (sm_int48_0[col_idx])
            );
            twos_sm_convert #(
                .DW(48)
            ) twos_sm_convert_int48_1 (
                .twos_complement_in(mul_sf_man_1[col_idx]),
                .signed_mag_out    (sm_int48_1[col_idx])
            );
            always_ff @(posedge clk) begin
                sm_mul_sf_man_0[col_idx] <= mul_sf_man_0[col_idx][47] ? sm_int48_0[col_idx] : mul_sf_man_0[col_idx];
                sm_mul_sf_man_1[col_idx] <= mul_sf_man_1[col_idx][47] ? sm_int48_1[col_idx] : mul_sf_man_1[col_idx];
                sm_sign_sf_man_0[col_idx] <= mul_sf_man_0[col_idx][47];
                sm_sign_sf_man_1[col_idx] <= mul_sf_man_1[col_idx][47];
            end
        end
    endgenerate

    // then, we should measure the leading zero/one, depending on the sign bit
    logic [COLS-1:0][5:0] right_shift_bits_0;
    logic [COLS-1:0][5:0] right_shift_bits_1;
    generate
        for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            lzc_48b lzc_48b_pos_0 (
                .a               (sm_mul_sf_man_0[col_idx]),
                .right_shift_bits(right_shift_bits_0[col_idx])
            );
            lzc_48b lzc_48b_pos_1 (
                .a               (sm_mul_sf_man_1[col_idx]),
                .right_shift_bits(right_shift_bits_1[col_idx])
            );
        end
    endgenerate

    // in the second cycle, calculate the exp and normed man of quantized fp16 numbers
    logic [COLS-1:0][7:0] exp_int8_fp16_0;
    logic [COLS-1:0][7:0] exp_int8_fp16_1;
    logic [COLS-1:0][6:0] man_int8_fp16_0;
    logic [COLS-1:0][6:0] man_int8_fp16_1;
    logic [COLS-1:0]      sign_int8_fp16_0;
    logic [COLS-1:0]      sign_int8_fp16_1;

    generate
        for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            always_ff @(posedge clk) begin
                exp_int8_fp16_0[col_idx]  <= right_shift_bits_0[col_idx] + sf_exp_r;
                exp_int8_fp16_1[col_idx]  <= right_shift_bits_1[col_idx] + sf_exp_r;
                man_int8_fp16_0[col_idx]  <= sm_mul_sf_man_0[col_idx][15:9];
                man_int8_fp16_1[col_idx]  <= sm_mul_sf_man_1[col_idx][15:9];
                sign_int8_fp16_0[col_idx] <= sm_sign_sf_man_0[col_idx];
                sign_int8_fp16_1[col_idx] <= sm_sign_sf_man_1[col_idx];
            end
        end
    endgenerate

    // now we have 512-bit fp16 data (i.e., 32 fp16 numbers), we should separate them into two 256-bit data
    logic                              int8_fp16_valid;
    logic                              int8_fp16_valid_temp;
    logic [AXIS_S2MM_DATA_WIDTH*2-1:0] int8_fp16_data_temp;
    logic [  AXIS_S2MM_DATA_WIDTH-1:0] int8_fp16_data_out;
    logic                              int8_fp16_last;

    // @TODO: parameterize
    assign int8_fp16_data_temp = {
        sign_int8_fp16_1[15],
        exp_int8_fp16_1[15],
        man_int8_fp16_1[15],
        sign_int8_fp16_0[15],
        exp_int8_fp16_0[15],
        man_int8_fp16_0[15],
        sign_int8_fp16_1[14],
        exp_int8_fp16_1[14],
        man_int8_fp16_1[14],
        sign_int8_fp16_0[14],
        exp_int8_fp16_0[14],
        man_int8_fp16_0[14],
        sign_int8_fp16_1[13],
        exp_int8_fp16_1[13],
        man_int8_fp16_1[13],
        sign_int8_fp16_0[13],
        exp_int8_fp16_0[13],
        man_int8_fp16_0[13],
        sign_int8_fp16_1[12],
        exp_int8_fp16_1[12],
        man_int8_fp16_1[12],
        sign_int8_fp16_0[12],
        exp_int8_fp16_0[12],
        man_int8_fp16_0[12],
        sign_int8_fp16_1[11],
        exp_int8_fp16_1[11],
        man_int8_fp16_1[11],
        sign_int8_fp16_0[11],
        exp_int8_fp16_0[11],
        man_int8_fp16_0[11],
        sign_int8_fp16_1[10],
        exp_int8_fp16_1[10],
        man_int8_fp16_1[10],
        sign_int8_fp16_0[10],
        exp_int8_fp16_0[10],
        man_int8_fp16_0[10],
        sign_int8_fp16_1[9],
        exp_int8_fp16_1[9],
        man_int8_fp16_1[9],
        sign_int8_fp16_0[9],
        exp_int8_fp16_0[9],
        man_int8_fp16_0[9],
        sign_int8_fp16_1[8],
        exp_int8_fp16_1[8],
        man_int8_fp16_1[8],
        sign_int8_fp16_0[8],
        exp_int8_fp16_0[8],
        man_int8_fp16_0[8],
        sign_int8_fp16_1[7],
        exp_int8_fp16_1[7],
        man_int8_fp16_1[7],
        sign_int8_fp16_0[7],
        exp_int8_fp16_0[7],
        man_int8_fp16_0[7],
        sign_int8_fp16_1[6],
        exp_int8_fp16_1[6],
        man_int8_fp16_1[6],
        sign_int8_fp16_0[6],
        exp_int8_fp16_0[6],
        man_int8_fp16_0[6],
        sign_int8_fp16_1[5],
        exp_int8_fp16_1[5],
        man_int8_fp16_1[5],
        sign_int8_fp16_0[5],
        exp_int8_fp16_0[5],
        man_int8_fp16_0[5],
        sign_int8_fp16_1[4],
        exp_int8_fp16_1[4],
        man_int8_fp16_1[4],
        sign_int8_fp16_0[4],
        exp_int8_fp16_0[4],
        man_int8_fp16_0[4],
        sign_int8_fp16_1[3],
        exp_int8_fp16_1[3],
        man_int8_fp16_1[3],
        sign_int8_fp16_0[3],
        exp_int8_fp16_0[3],
        man_int8_fp16_0[3],
        sign_int8_fp16_1[2],
        exp_int8_fp16_1[2],
        man_int8_fp16_1[2],
        sign_int8_fp16_0[2],
        exp_int8_fp16_0[2],
        man_int8_fp16_0[2],
        sign_int8_fp16_1[1],
        exp_int8_fp16_1[1],
        man_int8_fp16_1[1],
        sign_int8_fp16_0[1],
        exp_int8_fp16_0[1],
        man_int8_fp16_0[1],
        sign_int8_fp16_1[0],
        exp_int8_fp16_1[0],
        man_int8_fp16_1[0],
        sign_int8_fp16_0[0],
        exp_int8_fp16_0[0],
        man_int8_fp16_0[0]
    };

    delay_chain #(
        .DW (1),
        .LEN(2)
    ) delay_int8_fp16_valid (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (quant_data_in_valid),
        .out  (int8_fp16_valid_temp)
    );

    logic                              fifo_int8_fp16_rd_en;
    logic                              fifo_int8_fp16_empty;
    logic                              fifo_int8_fp16_full;
    logic [AXIS_S2MM_DATA_WIDTH*2-1:0] fifo_int8_fp16_dout;
    fifo_common #(
        .WR_DATA_WIDTH(AXIS_S2MM_DATA_WIDTH * 2),
        .FIFO_DEPTH   (256)
    ) fifo_int8_fp16 (
        .clk  (clk),
        .rst_n(rst_n),
        .wr_en(int8_fp16_valid_temp),
        .rd_en(fifo_int8_fp16_rd_en),
        .empty(fifo_int8_fp16_empty),
        .full (fifo_int8_fp16_full),
        .din  (int8_fp16_data_temp),
        .dout (fifo_int8_fp16_dout)
    );

    logic       int8_fp16_convert_state;
    logic [7:0] cnt_out_int8_fp16;
    logic       cnt_flip_int8_fp16;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            int8_fp16_convert_state <= 1'b0;
        end else if (~int8_fp16_convert_state & int8_fp16_valid_temp) begin
            int8_fp16_convert_state <= 1'b1;
        end else if (int8_fp16_convert_state & (cnt_out_int8_fp16 == quant_int8_fp16_depth)) begin
            int8_fp16_convert_state <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            cnt_out_int8_fp16 <= 0;
        end else if (int8_fp16_convert_state) begin
            if (cnt_out_int8_fp16 == quant_int8_fp16_depth) begin
                cnt_out_int8_fp16 <= 0;
            end else cnt_out_int8_fp16 <= cnt_out_int8_fp16 + 1;
        end
    end

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            cnt_flip_int8_fp16 <= 1'b0;
        end else if (int8_fp16_convert_state) begin
            cnt_flip_int8_fp16 <= ~cnt_flip_int8_fp16;
        end
    end

    assign fifo_int8_fp16_rd_en = int8_fp16_convert_state & (~cnt_flip_int8_fp16);

    assign int8_fp16_data_out = cnt_flip_int8_fp16 ? fifo_int8_fp16_dout[AXIS_S2MM_DATA_WIDTH-1:0] : fifo_int8_fp16_dout[AXIS_S2MM_DATA_WIDTH*2-1:AXIS_S2MM_DATA_WIDTH];
    always_ff @(posedge clk) begin : blockName
        int8_fp16_valid <= int8_fp16_convert_state;
        int8_fp16_last  <= int8_fp16_convert_state & (cnt_out_int8_fp16 == quant_int8_fp16_depth);
    end
    /************************************** int8 -> fp16 **************************************/

    /************************************** fp16 -> int8 **************************************/
    logic [COLS-1:0][7:0] fp16_int8_quant;
    // for fp16 -> int8 mode, the quantization is just truncate, like int8 -> int8
    // @NOTE: the host should carefully calculate the scaling factors for fp16 -> int8 mode
    // exp = 22- real_exp

    generate
        for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            always_ff @(posedge clk) begin
                fp16_int8_quant[col_idx] <= shift_sf_0[col_idx][7:0];
            end
        end
    endgenerate

    logic                            fp16_int8_valid;
    logic [AXIS_S2MM_DATA_WIDTH-1:0] fp16_int8_data_out;
    logic                            fp16_int8_last;

    // the MSB 128-bit data is empty
    assign fp16_int8_data_out = {
        128'd0,
        fp16_int8_quant[15],
        fp16_int8_quant[14],
        fp16_int8_quant[13],
        fp16_int8_quant[12],
        fp16_int8_quant[11],
        fp16_int8_quant[10],
        fp16_int8_quant[9],
        fp16_int8_quant[8],
        fp16_int8_quant[7],
        fp16_int8_quant[6],
        fp16_int8_quant[5],
        fp16_int8_quant[4],
        fp16_int8_quant[3],
        fp16_int8_quant[2],
        fp16_int8_quant[1],
        fp16_int8_quant[0]
    };

    // valid signal is 2-cycle delayed
    delay_chain #(
        .DW (1),
        .LEN(2)
    ) delay_fp16_int8_valid (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (quant_data_in_valid),
        .out  (fp16_int8_valid)
    );

    delay_chain #(
        .DW (1),
        .LEN(2)
    ) delay_fp16_int8_last (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (quant_data_in_last),
        .out  (fp16_int8_last)
    );
    /************************************** fp16 -> int8 **************************************/

    /************************************** fp16 -> fp16 **************************************/
    // there is nothing to do for fp16 -> fp16 mode, just pass the data
    logic                            fp16_fp16_valid;
    logic [AXIS_S2MM_DATA_WIDTH-1:0] fp16_fp16_data_out;
    logic                            fp16_fp16_last;

    assign fp16_fp16_valid    = quant_data_in_valid;
    assign fp16_fp16_last     = quant_data_in_last;
    assign fp16_fp16_data_out = quant_data_in;
    /************************************** fp16 -> fp16 **************************************/

    /*************************************** Output MUX ***************************************/
    // quant mode. 00: int8 -> int8, 01: int8 -> fp16, 10: fp16 -> int8, 11: fp16 -> fp16
    always_ff @(posedge clk) begin
        if (quant_mode == 2'b00) begin
            quant_s2mm_out       <= int8_int8_data_out;
            quant_s2mm_out_valid <= int8_int8_valid;
            quant_s2mm_out_last  <= int8_int8_last;
        end else if (quant_mode == 2'b01) begin
            quant_s2mm_out       <= int8_fp16_data_out;
            quant_s2mm_out_valid <= int8_fp16_valid;
            quant_s2mm_out_last  <= int8_fp16_last;
        end else if (quant_mode == 2'b10) begin
            quant_s2mm_out       <= fp16_int8_data_out;
            quant_s2mm_out_valid <= fp16_int8_valid;
            quant_s2mm_out_last  <= fp16_int8_last;
        end else begin
            quant_s2mm_out       <= fp16_fp16_data_out;
            quant_s2mm_out_valid <= fp16_fp16_valid;
            quant_s2mm_out_last  <= fp16_fp16_last;
        end
    end
    /*************************************** Output MUX ***************************************/

endmodule
