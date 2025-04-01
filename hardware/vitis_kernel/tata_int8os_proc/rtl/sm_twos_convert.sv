`timescale 1ns / 1ps

// convert a 9-bit signed magnitude number to a 9-bit two's complement number
module sm_twos_convert (
    input  logic        [8:0] signed_mag_in,
    output logic signed [8:0] twos_complement_out
);

    // note that the signed_mag_in is not signed representation in 2's complement
    logic [7:0] mag_part;
    logic [7:0] reverse_mag_part;
    logic       sign_bit;
    assign mag_part         = signed_mag_in[7:0];
    assign sign_bit         = signed_mag_in[8];
    assign reverse_mag_part = (~mag_part) + 1;

    assign twos_complement_out = sign_bit ? {1'b1, reverse_mag_part} : signed_mag_in;

endmodule
