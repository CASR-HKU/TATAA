`timescale 1ns / 1ps

module pe_stg_0 #(
    parameter LEFT_WIDTH   = 8,
    parameter RIGHT_WIDTH  = 8,
    parameter TOP_WIDTH    = 48,
    parameter BOTTOM_WIDTH = 48,
    parameter PRELD_WIDTH  = 16
) (
    // common signals
    input  logic                    clk,
    input  logic                    rst_n,
    // y_sel_in is used to indicate Y tile.
    input  logic                    y_sel_in,
    // store_en is used output
    input  logic                    sys_buf_en_in,
    // 00: matrix multiplication, 10: fp mul, 11: fp add
    input  logic [             1:0] mode_sel_in,
    // psu clear
    input  logic                    psu_clr_in,
    // systolic array signals
    input  logic [  LEFT_WIDTH-1:0] left_in,
    output logic [ RIGHT_WIDTH-1:0] right_out,
    input  logic [   TOP_WIDTH-1:0] top_in,
    output logic [BOTTOM_WIDTH-1:0] bottom_out
);

    // DSP reset
    logic rst = 1'b0;
    always_ff @(posedge clk) begin
        rst <= ~rst_n;
    end

    // preload Y regs
    logic [PRELD_WIDTH-1:0] y_reg;
    assign y_reg = y_sel_in ? top_in[31:16] : top_in[15:0];

    // left to right
    always_ff @(posedge clk) begin
        right_out <= left_in;
    end

    // DSP
    logic [29:0] dsp_a_in;
    logic [17:0] dsp_b_in;
    logic [47:0] dsp_c_in;
    logic [26:0] dsp_d_in;
    logic [47:0] dsp_p_out;
    logic [47:0] psu_acc_reg;

    always_comb begin
        if (mode_sel_in == 2'b00) begin
            dsp_a_in = {{(3) {y_reg[15]}}, y_reg[15:8], 19'd0};
        end else begin
            dsp_a_in = {22'd0, top_in[14:7]};
        end
    end

    always_comb begin
        if (mode_sel_in == 2'b00) begin
            dsp_b_in = {{(10) {left_in[7]}}, left_in};
        end else begin
            dsp_b_in = 18'd1;
        end
    end

    generate
        always_comb begin
            if (mode_sel_in == 2'b10) begin
                dsp_c_in = -48'sd127;
            end else begin
                dsp_c_in = psu_acc_reg;
            end
        end
    endgenerate


    always_comb begin
        if (mode_sel_in == 2'b00) begin
            dsp_d_in = {{(19) {y_reg[7]}}, y_reg[7:0]};
        end else begin
            dsp_d_in = {19'd0, top_in[30:23]};
        end
    end

    logic [4:0] dsp_in_mode;
    always_comb begin
        if (mode_sel_in == 2'b11) begin
            dsp_in_mode = 5'b11101;
        end else begin
            dsp_in_mode = 5'b10101;
        end
    end

    logic [8:0] ext_man_0;
    logic [8:0] ext_man_1;
    logic [8:0] twos_man_0;
    logic [8:0] twos_man_1;
    logic [8:0] twos_man_0_sync;
    logic [8:0] twos_man_1_sync;
    assign ext_man_0 = {top_in[15], 1'b1, top_in[6:0]};
    assign ext_man_1 = {top_in[31], 1'b1, top_in[22:16]};

    sm_twos_convert sm_twos_man_0 (
        .signed_mag_in(ext_man_0),
        .twos_complement_out(twos_man_0)
    );

    sm_twos_convert sm_twos_man_1 (
        .signed_mag_in(ext_man_1),
        .twos_complement_out(twos_man_1)
    );

    delay_chain #(
        .DW (9),
        .LEN(3)
    ) delay_chain_man_0 (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (twos_man_0),
        .out  (twos_man_0_sync)
    );

    delay_chain #(
        .DW (9),
        .LEN(3)
    ) delay_chain_man_1 (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (twos_man_1),
        .out  (twos_man_1_sync)
    );

    logic [7:0] exp_0_sync;
    logic [7:0] exp_1_sync;

    delay_chain #(
        .DW (8),
        .LEN(3)
    ) delay_chain_exp_0 (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (top_in[14:7]),
        .out  (exp_0_sync)
    );

    delay_chain #(
        .DW (8),
        .LEN(3)
    ) delay_chain_exp_1 (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (1'b1),
        .in   (top_in[30:23]),
        .out  (exp_1_sync)
    );

    DSP48E2 #(
        // Feature Control Attributes: Data Path Selection
        .AMULTSEL("AD"),  // Selects A input to multiplier (A, AD)
        .A_INPUT("DIRECT"),  // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
        .BMULTSEL("B"),  // Selects B input to multiplier (AD, B)
        .B_INPUT("DIRECT"),  // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
        .PREADDINSEL("A"),  // Selects input to pre-adder (A, B)
        .RND(48'h000000000000),  // Rounding Constant
        .USE_MULT("MULTIPLY"),  // Select multiplier usage (DYNAMIC, MULTIPLY, NONE)
        .USE_SIMD("ONE48"),  // SIMD selection (FOUR12, ONE48, TWO24)
        .USE_WIDEXOR("FALSE"),  // Use the Wide XOR function (FALSE, TRUE)
        .XORSIMD("XOR24_48_96"),  // Mode of operation for the Wide XOR (XOR12, XOR24_48_96)
        // Pattern Detector Attributes: Pattern Detection Configuration
        .AUTORESET_PATDET("NO_RESET"),  // NO_RESET, RESET_MATCH, RESET_NOT_MATCH
        .AUTORESET_PRIORITY("RESET"),  // Priority of AUTORESET vs. CEP (CEP, RESET).
        .MASK(48'h3fffffffffff),  // 48-bit mask value for pattern detect (1=ignore)
        .PATTERN(48'h000000000000),  // 48-bit pattern match for pattern detect
        .SEL_MASK("MASK"),  // C, MASK, ROUNDING_MODE1, ROUNDING_MODE2
        .SEL_PATTERN("PATTERN"),  // Select pattern value (C, PATTERN)
        .USE_PATTERN_DETECT("NO_PATDET"),  // Enable pattern detect (NO_PATDET, PATDET)
        // Programmable Inversion Attributes: Specifies built-in programmable inversion on specific pins
        .IS_ALUMODE_INVERTED(4'b0000),  // Optional inversion for ALUMODE
        .IS_CARRYIN_INVERTED(1'b0),  // Optional inversion for CARRYIN
        .IS_CLK_INVERTED(1'b0),  // Optional inversion for CLK
        .IS_INMODE_INVERTED(5'b00000),  // Optional inversion for INMODE
        .IS_OPMODE_INVERTED(9'b000000000),  // Optional inversion for OPMODE
        .IS_RSTALLCARRYIN_INVERTED(1'b0),  // Optional inversion for RSTALLCARRYIN
        .IS_RSTALUMODE_INVERTED(1'b0),  // Optional inversion for RSTALUMODE
        .IS_RSTA_INVERTED(1'b0),  // Optional inversion for RSTA
        .IS_RSTB_INVERTED(1'b0),  // Optional inversion for RSTB
        .IS_RSTCTRL_INVERTED(1'b0),  // Optional inversion for RSTCTRL
        .IS_RSTC_INVERTED(1'b0),  // Optional inversion for RSTC
        .IS_RSTD_INVERTED(1'b0),  // Optional inversion for RSTD
        .IS_RSTINMODE_INVERTED(1'b0),  // Optional inversion for RSTINMODE
        .IS_RSTM_INVERTED(1'b0),  // Optional inversion for RSTM
        .IS_RSTP_INVERTED(1'b0),  // Optional inversion for RSTP
        // Register Control Attributes: Pipeline Register Configuration
        .ACASCREG(1),  // Number of pipeline stages between A/ACIN and ACOUT (0-2)
        .ADREG(0),  // Pipeline stages for pre-adder (0-1)
        .ALUMODEREG(0),  // Pipeline stages for ALUMODE (0-1)
        .AREG(1),  // Pipeline stages for A (0-2)
        .BCASCREG(1),  // Number of pipeline stages between B/BCIN and BCOUT (0-2)
        .BREG(1),  // Pipeline stages for B (0-2)
        .CARRYINREG(1),  // Pipeline stages for CARRYIN (0-1)
        .CARRYINSELREG(1),  // Pipeline stages for CARRYINSEL (0-1)
        .CREG(0),  // Pipeline stages for C (0-1)
        .DREG(1),  // Pipeline stages for D (0-1)
        .INMODEREG(0),  // Pipeline stages for INMODE (0-1)
        .MREG(1),  // Multiplier pipeline stages (0-1)
        .OPMODEREG(0),  // Pipeline stages for OPMODE (0-1)
        .PREG(0)  // Number of pipeline stages for P (0-1)
    ) DSP48E2_inst (
        // Cascade outputs: Cascade Ports
        // .ACOUT(ACOUT),                   // 30-bit output: A port cascade
        // .BCOUT(BCOUT),                   // 18-bit output: B cascade
        // .CARRYCASCOUT(CARRYCASCOUT),     // 1-bit output: Cascade carry
        // .MULTSIGNOUT(MULTSIGNOUT),       // 1-bit output: Multiplier sign cascade
        // .PCOUT(PCOUT),                   // 48-bit output: Cascade output
        // Control outputs: Control Inputs/Status Bits
        // .OVERFLOW(OVERFLOW),             // 1-bit output: Overflow in add/acc
        // .PATTERNBDETECT(PATTERNBDETECT), // 1-bit output: Pattern bar detect
        // .PATTERNDETECT(PATTERNDETECT),   // 1-bit output: Pattern detect
        // .UNDERFLOW(UNDERFLOW),           // 1-bit output: Underflow in add/acc
        // Data outputs: Data Ports
        // .CARRYOUT(CARRYOUT),             // 4-bit output: Carry
        .P(dsp_p_out),  // 48-bit output: Primary data
        // .XOROUT(XOROUT),                 // 8-bit output: XOR data
        // Cascade inputs: Cascade Ports
        // .ACIN(ACIN),                     // 30-bit input: A cascade data
        // .BCIN(BCIN),                     // 18-bit input: B cascade
        // .CARRYCASCIN(CARRYCASCIN),       // 1-bit input: Cascade carry
        // .MULTSIGNIN(MULTSIGNIN),         // 1-bit input: Multiplier sign cascade
        // .PCIN(dsp_pc_in),  // 48-bit input: P cascade
        // Control inputs: Control Inputs/Status Bits
        .ALUMODE(4'b0000),  // 4-bit input: ALU control
        // .CARRYINSEL(CARRYINSEL),         // 3-bit input: Carry select
        .CLK(clk),  // 1-bit input: Clock
        .INMODE(dsp_in_mode),  // 5-bit input: INMODE control
        .OPMODE(9'b00_011_01_01),  // 9-bit input: Operation mode
        // Data inputs: Data Ports
        .A(dsp_a_in),  // 30-bit input: A data
        .B(dsp_b_in),  // 18-bit input: B data
        .C(dsp_c_in),  // 48-bit input: C data
        // .CARRYIN(CARRYIN),               // 1-bit input: Carry-in
        .D(dsp_d_in),  // 27-bit input: D data
        // Reset/Clock Enable inputs: Reset/Clock Enable Inputs
        .CEA1(1'b1),  // 1-bit input: Clock enable for 1st stage AREG
        .CEA2(1'b1),  // 1-bit input: Clock enable for 2nd stage AREG
        .CEAD(1'b0),  // 1-bit input: Clock enable for ADREG
        .CEALUMODE(1'b0),  // 1-bit input: Clock enable for ALUMODE
        .CEB1(1'b1),  // 1-bit input: Clock enable for 1st stage BREG
        .CEB2(1'b1),  // 1-bit input: Clock enable for 2nd stage BREG
        .CEC(1'b1),  // 1-bit input: Clock enable for CREG
        .CECARRYIN(1'b0),  // 1-bit input: Clock enable for CARRYINREG
        .CECTRL(1'b0),  // 1-bit input: Clock enable for OPMODEREG and CARRYINSELREG
        .CED(1'b1),  // 1-bit input: Clock enable for DREG
        .CEINMODE(1'b0),  // 1-bit input: Clock enable for INMODEREG
        .CEM(1'b1),  // 1-bit input: Clock enable for MREG
        .CEP(1'b1),  // 1-bit input: Clock enable for PREG
        .RSTA(rst),  // 1-bit input: Reset for AREG
        .RSTALLCARRYIN(rst),  // 1-bit input: Reset for CARRYINREG
        .RSTALUMODE(1'b0),  // 1-bit input: Reset for ALUMODEREG
        .RSTB(rst),  // 1-bit input: Reset for BREG
        .RSTC(1'b0),  // 1-bit input: Reset for CREG
        .RSTCTRL(1'b0),  // 1-bit input: Reset for OPMODEREG and CARRYINSELREG
        .RSTD(1'b0),  // 1-bit input: Reset for DREG and ADREG
        .RSTINMODE(1'b0),  // 1-bit input: Reset for INMODEREG
        .RSTM(rst),  // 1-bit input: Reset for MREG
        .RSTP(rst)  // 1-bit input: Reset for PREG
    );

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            psu_acc_reg <= 0;
        end else if (psu_clr_in) begin
            psu_acc_reg <= 0;
        end else if (sys_buf_en_in) begin
            psu_acc_reg <= top_in;
        end else begin
            psu_acc_reg <= dsp_p_out;
        end
    end

    always_ff @(posedge clk) begin
        if (sys_buf_en_in) begin
            bottom_out <= psu_acc_reg;
        end else if (mode_sel_in == 2'b00) begin  // matrix multiplication
            bottom_out <= top_in;
        end else begin  // fp mul & add
            bottom_out <= {
                4'd0, exp_1_sync, exp_0_sync, dsp_p_out[9:0], twos_man_1_sync, twos_man_0_sync
            };
        end
    end

endmodule
