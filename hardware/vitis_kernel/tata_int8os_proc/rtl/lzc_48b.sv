`timescale 1ns / 1ps

// count the leading zeros of a 48-bit number
// then directly output 31 - lzc
// @TODO: paramterize and optimize
module lzc_48b (
    input  logic [47:0] a,
    output logic [ 5:0] right_shift_bits 
);
    always_comb begin
        if (a[47] == 1'b1) begin
            right_shift_bits = 31;
        end
        else if (a[46] == 1'b1) begin
            right_shift_bits = 30;
        end
        else if (a[45] == 1'b1) begin
            right_shift_bits = 29;
        end
        else if (a[44] == 1'b1) begin
            right_shift_bits = 28;
        end
        else if (a[43] == 1'b1) begin
            right_shift_bits = 27;
        end
        else if (a[42] == 1'b1) begin
            right_shift_bits = 26;
        end
        else if (a[41] == 1'b1) begin
            right_shift_bits = 25;
        end
        else if (a[40] == 1'b1) begin
            right_shift_bits = 24;
        end
        else if (a[39] == 1'b1) begin
            right_shift_bits = 23;
        end
        else if (a[38] == 1'b1) begin
            right_shift_bits = 22;
        end
        else if (a[37] == 1'b1) begin
            right_shift_bits = 21;
        end
        else if (a[36] == 1'b1) begin
            right_shift_bits = 20;
        end
        else if (a[35] == 1'b1) begin
            right_shift_bits = 19;
        end
        else if (a[34] == 1'b1) begin
            right_shift_bits = 18;
        end
        else if (a[33] == 1'b1) begin
            right_shift_bits = 17;
        end
        else if (a[32] == 1'b1) begin
            right_shift_bits = 16;
        end
        else if (a[31] == 1'b1) begin
            right_shift_bits = 15;
        end
        else if (a[30] == 1'b1) begin
            right_shift_bits = 14;
        end
        else if (a[29] == 1'b1) begin
            right_shift_bits = 13;
        end
        else if (a[28] == 1'b1) begin
            right_shift_bits = 12;
        end
        else if (a[27] == 1'b1) begin
            right_shift_bits = 11;
        end
        else if (a[26] == 1'b1) begin
            right_shift_bits = 10;
        end
        else if (a[25] == 1'b1) begin
            right_shift_bits = 9;
        end
        else if (a[24] == 1'b1) begin
            right_shift_bits = 8;
        end
        else if (a[23] == 1'b1) begin
            right_shift_bits = 7;
        end
        else if (a[22] == 1'b1) begin
            right_shift_bits = 6;
        end
        else if (a[21] == 1'b1) begin
            right_shift_bits = 5;
        end
        else if (a[20] == 1'b1) begin
            right_shift_bits = 4;
        end
        else if (a[19] == 1'b1) begin
            right_shift_bits = 3;
        end
        else if (a[18] == 1'b1) begin
            right_shift_bits = 2;
        end
        else if (a[17] == 1'b1) begin
            right_shift_bits = 1;
        end
        else if (a[16] == 1'b1) begin
            right_shift_bits = 0;
        end
        else begin
            right_shift_bits = 0;
        end
    end
endmodule