`timescale 1ns / 1ps

module delay_chain #(
    parameter DW  = 8,
    parameter LEN = 4
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              en,
    input  logic [DW - 1 : 0] in,
    output logic [DW - 1 : 0] out
);

    logic [DW - 1 : 0] dly[LEN : 0];

    genvar i, j;
    generate
        if (LEN == 0) begin
            assign out = in;
        end else if (LEN == 1) begin
            always_ff @(posedge clk) begin
                out <= in;
            end
        end else begin
            assign dly[0] = in;
            assign out    = dly[LEN];
            for (j = 0; j < DW; j = j + 1) begin
                for (i = 1; i <= LEN; i = i + 1) begin
                    FDRE DeFF (
                        .Q (dly[i][j]),
                        .C (clk),
                        .CE(en),
                        .D (dly[i-1][j]),
                        .R (~rst_n)
                    );
                end
            end
        end
    endgenerate

endmodule
