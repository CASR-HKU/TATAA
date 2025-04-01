`timescale 1ns / 1ps

// barrel right shifter, c = a >> b
// TODO: parameterized
module bs_rsf (
    input  logic signed [47:0] a,
    input  logic        [ 4:0] b,
    output logic signed [47:0] c
);

    always_comb begin
        case (b)
            5'd0:    c = a;
            5'd1:    c = a >>> 1;
            5'd2:    c = a >>> 2;
            5'd3:    c = a >>> 3;
            5'd4:    c = a >>> 4;
            5'd5:    c = a >>> 5;
            5'd6:    c = a >>> 6;
            5'd7:    c = a >>> 7;
            5'd8:    c = a >>> 8;
            5'd9:    c = a >>> 9;
            5'd10:   c = a >>> 10;
            5'd11:   c = a >>> 11;
            5'd12:   c = a >>> 12;
            5'd13:   c = a >>> 13;
            5'd14:   c = a >>> 14;
            5'd15:   c = a >>> 15;
            5'd16:   c = a >>> 16;
            5'd17:   c = a >>> 17;
            5'd18:   c = a >>> 18;
            5'd19:   c = a >>> 19;
            5'd20:   c = a >>> 20;
            5'd21:   c = a >>> 21;
            5'd22:   c = a >>> 22;
            5'd23:   c = a >>> 23;
            5'd24:   c = a >>> 24;
            5'd25:   c = a >>> 25;
            5'd26:   c = a >>> 26;
            5'd27:   c = a >>> 27;
            5'd28:   c = a >>> 28;
            5'd29:   c = a >>> 29;
            5'd30:   c = a >>> 30;
            5'd31:   c = a >>> 31;
            default: c = 48'd0;
        endcase
    end

endmodule
