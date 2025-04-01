`timescale 1ns / 1ps

module pccmd_ctrl #(
    parameter AXIS_PCCMD_DATA_WIDTH = 32,
    parameter AXIS_PCFBK_DATA_WIDTH = 8,
    parameter PSU_DEPTH_WIDTH       = 9,
    parameter DMRFY_ADDR_WIDTH      = 5,
    parameter QUANT_SCALE_EXP_WIDTH = 5,
    parameter QUANT_SCALE_MAN_WIDTH = 16
) (
    input  logic                             clk,
    input  logic                             rst_n,
    // pccmd axis
    input  logic                             s_axis_pccmd_tvalid,
    output logic                             s_axis_pccmd_tready,
    input  logic [AXIS_PCCMD_DATA_WIDTH-1:0] s_axis_pccmd_tdata,
    // feedback axis
    output logic                             m_axis_pcfbk_tvalid,
    input  logic                             m_axis_pcfbk_tready,
    output logic [AXIS_PCFBK_DATA_WIDTH-1:0] m_axis_pcfbk_tdata,
    // parameters and control signals
    output logic                             pc_load_mode_sel,
    output logic [                      1:0] pc_exec_mode_sel,
    output logic [                      1:0] pc_store_mode_sel,
    output logic [      PSU_DEPTH_WIDTH-1:0] psu_fpv_depth,
    output logic [      PSU_DEPTH_WIDTH-1:0] xi_load_depth,
    output logic [      PSU_DEPTH_WIDTH-1:0] yi_load_depth,
    output logic [                      6:0] store_depth,
    output logic [                      3:0] store_tapu_depth,
    output logic                             psu_clr,
    output logic                             psu_acc,
    output logic                             load_tile_sel,
    output logic                             exec_tile_sel,
    output logic [     DMRFY_ADDR_WIDTH-1:0] pc_fp_load_vec,
    output logic                             pc_fp_load_sel,
    output logic                             pc_fp_updt_en,
    output logic                             pc_fp_updt_sel,
    output logic [     DMRFY_ADDR_WIDTH-1:0] pc_fp_updt_vec,
    output logic [     DMRFY_ADDR_WIDTH-1:0] pc_fp0_exec_vec,
    output logic [     DMRFY_ADDR_WIDTH-1:0] pc_fp1_exec_vec,
    output logic [QUANT_SCALE_EXP_WIDTH-1:0] pc_quant_sf_exp,
    output logic [QUANT_SCALE_MAN_WIDTH-1:0] pc_quant_sf_man,
    output logic                             pc_exec_start,
    output logic                             pc_store_start,
    input  logic                             pc_loadx_done,
    input  logic                             pc_loady_done,
    input  logic                             pc_exec_done,
    input  logic                             pc_store_done
);

    logic [2:0] pccmd_type;
    logic       hs_axis_pccmd;
    assign pccmd_type    = s_axis_pccmd_tdata[2:0];
    assign hs_axis_pccmd = s_axis_pccmd_tvalid & s_axis_pccmd_tready;

    // pc_exec_mode_sel is only updated in load commands
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            pc_load_mode_sel  <= 0;
            load_tile_sel     <= 0;
            xi_load_depth     <= 0;
            yi_load_depth     <= 0;
            pc_fp_load_sel    <= 0;
            pc_fp_load_vec    <= 0;
            pc_exec_mode_sel  <= 0;
            exec_tile_sel     <= 0;
            psu_fpv_depth     <= 0;
            psu_acc           <= 0;
            psu_clr           <= 0;
            pc_fp_updt_en     <= 0;
            pc_fp_updt_sel    <= 0;
            pc_fp0_exec_vec   <= 0;
            pc_fp1_exec_vec   <= 0;
            pc_fp_updt_vec    <= 0;
            pc_store_mode_sel <= 0;
            store_depth       <= 0;
            pc_quant_sf_exp   <= 0;
            pc_quant_sf_man   <= 0;
        end else if (hs_axis_pccmd & (pccmd_type == 3'b000)) begin  // configuration
            pc_quant_sf_exp <= s_axis_pccmd_tdata[7:3];
            pc_quant_sf_man <= s_axis_pccmd_tdata[23:8];
            pc_load_mode_sel <= s_axis_pccmd_tdata[24];
        end else if (hs_axis_pccmd & (pccmd_type == 3'b001)) begin  // load int8
            load_tile_sel    <= s_axis_pccmd_tdata[4];
            xi_load_depth    <= s_axis_pccmd_tdata[14:6];
            yi_load_depth    <= s_axis_pccmd_tdata[14:6];
        end else if (hs_axis_pccmd & (pccmd_type == 3'b010)) begin  // load fpv
            pc_fp_load_sel   <= s_axis_pccmd_tdata[4];
            pc_fp_load_vec   <= s_axis_pccmd_tdata[10:6];
        end else if (hs_axis_pccmd & (pccmd_type == 3'b100)) begin  // exec blk
            pc_exec_mode_sel <= {pc_load_mode_sel, s_axis_pccmd_tdata[3]};
            exec_tile_sel    <= s_axis_pccmd_tdata[4];
            psu_fpv_depth    <= s_axis_pccmd_tdata[14:6];
            psu_acc          <= s_axis_pccmd_tdata[15];
            psu_clr          <= s_axis_pccmd_tdata[16];
        end else if (hs_axis_pccmd & (pccmd_type == 3'b101)) begin  // exec fpv
            pc_exec_mode_sel <= {pc_load_mode_sel, s_axis_pccmd_tdata[3]};
            pc_fp_updt_en    <= s_axis_pccmd_tdata[4];
            pc_fp_updt_sel   <= s_axis_pccmd_tdata[5];
            pc_fp0_exec_vec  <= s_axis_pccmd_tdata[10:6];
            pc_fp1_exec_vec  <= s_axis_pccmd_tdata[15:11];
            pc_fp_updt_vec   <= s_axis_pccmd_tdata[20:16];
        end else if (hs_axis_pccmd & (pccmd_type == 3'b011)) begin  // store
            pc_store_mode_sel <= s_axis_pccmd_tdata[4:3];
            store_depth       <= s_axis_pccmd_tdata[12:6];
            store_tapu_depth  <= s_axis_pccmd_tdata[16:13];
        end
    end

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            pc_exec_start  <= 1'b0;
            pc_store_start <= 1'b0;
        end else begin
            pc_exec_start  <= hs_axis_pccmd & (pccmd_type[2] == 1'b1);
            pc_store_start <= hs_axis_pccmd & (pccmd_type == 3'b011);
        end
    end

    logic done_state;
    assign done_state = pc_loadx_done | pc_loady_done | pc_exec_done | pc_store_done;

    logic pcfbk_en;
    assign pcfbk_en = done_state & ((~m_axis_pcfbk_tvalid) | m_axis_pcfbk_tready);

    logic hs_axis_pcfbk;
    assign hs_axis_pcfbk = m_axis_pcfbk_tvalid & m_axis_pcfbk_tready;

    logic pcfbk_valid_r;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            pcfbk_valid_r <= 1'b0;
        end else if (pcfbk_en) begin
            pcfbk_valid_r <= 1'b1;
        end else if (hs_axis_pcfbk & (~pcfbk_en)) begin
            pcfbk_valid_r <= 1'b0;
        end
    end
    assign m_axis_pcfbk_tvalid = pcfbk_valid_r;

    always_ff @(posedge clk) begin
        m_axis_pcfbk_tdata <= {4'd0, pc_loady_done, pc_loadx_done, pc_exec_done, pc_store_done};
    end

    assign s_axis_pccmd_tready = 1'b1;

endmodule
