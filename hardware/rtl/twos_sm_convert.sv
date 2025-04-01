`timescale 1ns / 1ps

// convert a 9-bit two's complement number to a 9-bit signed magnitude number
module twos_sm_convert #(
    parameter DW = 9
) (
    input  logic signed [DW-1:0] twos_complement_in,
    output logic        [DW-1:0] signed_mag_out
);

    logic [DW-2:0] twos_part;
    logic [DW-2:0] reverse_twos_part;
    logic       sign_bit;
    assign twos_part         = twos_complement_in[DW-2:0];
    assign sign_bit          = twos_complement_in[DW-1];
    assign reverse_twos_part = (~twos_part) + 1;

    assign signed_mag_out = sign_bit ? {1'b1, reverse_twos_part} : twos_complement_in;

endmodule
