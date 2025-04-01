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
    output logic [                      7:0] store_depth,
    output logic [                      4:0] store_tapu_depth,
    output logic                             store_mt_en,
    output logic                             psu_clr,
    output logic                             psu_acc,
    output logic                             loadx_tile_sel,
    output logic                             execx_tile_sel,
    output logic                             loady_tile_sel,
    output logic                             execy_tile_sel,
    output logic [                     15:0] const_fp16_reg,
    output logic [     DMRFY_ADDR_WIDTH-1:0] pc_fp0_load_vec,
    output logic [     DMRFY_ADDR_WIDTH-1:0] pc_fp1_load_vec,
    output logic                             pc_fp_updt_en,
    output logic                             pc_fp_updt_sel,
    output logic [     DMRFY_ADDR_WIDTH-1:0] pc_fp_updt_vec,
    output logic [     DMRFY_ADDR_WIDTH-1:0] pc_fp0_exec_vec,
    output logic [     DMRFY_ADDR_WIDTH-1:0] pc_fp1_exec_vec,
    output logic [2:0]                       pc_fp_op_sel,
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

    logic [7:0][15:0] const_fp16_reg_files;
    logic [2:0]       const_fp16_reg_files_sel;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            const_fp16_reg <= 0;
        end else begin
            case (const_fp16_reg_files_sel)
                3'b000:  const_fp16_reg <= const_fp16_reg_files[0];
                3'b001:  const_fp16_reg <= const_fp16_reg_files[1];
                3'b010:  const_fp16_reg <= const_fp16_reg_files[2];
                3'b011:  const_fp16_reg <= const_fp16_reg_files[3];
                3'b100:  const_fp16_reg <= const_fp16_reg_files[4];
                3'b101:  const_fp16_reg <= const_fp16_reg_files[5];
                3'b110:  const_fp16_reg <= const_fp16_reg_files[6];
                3'b111:  const_fp16_reg <= const_fp16_reg_files[7];
                default: const_fp16_reg <= const_fp16_reg_files[0];
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            pc_load_mode_sel         <= 0;
            loadx_tile_sel           <= 0;
            loady_tile_sel           <= 0;
            xi_load_depth            <= 0;
            yi_load_depth            <= 0;
            pc_fp0_load_vec          <= 0;
            pc_fp1_load_vec          <= 0;
            pc_exec_mode_sel         <= 0;
            execx_tile_sel           <= 0;
            execy_tile_sel           <= 0;
            psu_fpv_depth            <= 0;
            psu_acc                  <= 0;
            psu_clr                  <= 0;
            pc_fp_updt_en            <= 0;
            pc_fp_updt_sel           <= 0;
            pc_fp0_exec_vec          <= 0;
            pc_fp1_exec_vec          <= 0;
            pc_fp_updt_vec           <= 0;
            pc_store_mode_sel        <= 0;
            store_depth              <= 0;
            store_mt_en              <= 0;
            pc_quant_sf_exp          <= 0;
            pc_quant_sf_man          <= 0;
            pc_fp_op_sel             <= 0;
            const_fp16_reg_files_sel <= 0;
            const_fp16_reg_files[0]  <= 0;
            const_fp16_reg_files[1]  <= 0;
            const_fp16_reg_files[2]  <= 0;
            const_fp16_reg_files[3]  <= 0;
            const_fp16_reg_files[4]  <= 0;
            const_fp16_reg_files[5]  <= 0;
            const_fp16_reg_files[6]  <= 0;
            const_fp16_reg_files[7]  <= 0;
        end else if (hs_axis_pccmd & (pccmd_type == 3'b000)) begin  // configuration
            if (s_axis_pccmd_tdata[6] == 1'b1) begin
                pc_quant_sf_exp <= s_axis_pccmd_tdata[11:7];
                pc_quant_sf_man <= s_axis_pccmd_tdata[27:12];
            end else if (s_axis_pccmd_tdata[5:3] == 3'b000) begin
                const_fp16_reg_files[0] <= s_axis_pccmd_tdata[22:7];
            end else if (s_axis_pccmd_tdata[5:3] == 3'b001) begin
                const_fp16_reg_files[1] <= s_axis_pccmd_tdata[22:7];
            end else if (s_axis_pccmd_tdata[5:3] == 3'b010) begin
                const_fp16_reg_files[2] <= s_axis_pccmd_tdata[22:7];
            end else if (s_axis_pccmd_tdata[5:3] == 3'b011) begin
                const_fp16_reg_files[3] <= s_axis_pccmd_tdata[22:7];
            end else if (s_axis_pccmd_tdata[5:3] == 3'b100) begin
                const_fp16_reg_files[4] <= s_axis_pccmd_tdata[22:7];
            end else if (s_axis_pccmd_tdata[5:3] == 3'b101) begin
                const_fp16_reg_files[5] <= s_axis_pccmd_tdata[22:7];
            end else if (s_axis_pccmd_tdata[5:3] == 3'b110) begin
                const_fp16_reg_files[6] <= s_axis_pccmd_tdata[22:7];
            end else if (s_axis_pccmd_tdata[5:3] == 3'b111) begin
                const_fp16_reg_files[7] <= s_axis_pccmd_tdata[22:7];
            end
        end else if (hs_axis_pccmd & (pccmd_type == 3'b001)) begin  // load int8
            pc_load_mode_sel <= 1'b0;
            loadx_tile_sel   <= (~s_axis_pccmd_tdata[3]) ? s_axis_pccmd_tdata[4] : loadx_tile_sel;
            loady_tile_sel   <= s_axis_pccmd_tdata[3] ? s_axis_pccmd_tdata[4] : loady_tile_sel;
            xi_load_depth    <= (~s_axis_pccmd_tdata[3]) ? s_axis_pccmd_tdata[14:6] : xi_load_depth;
            yi_load_depth    <= s_axis_pccmd_tdata[3] ? s_axis_pccmd_tdata[14:6] : yi_load_depth;
        end else if (hs_axis_pccmd & (pccmd_type == 3'b010)) begin  // load fpv
            pc_load_mode_sel <= 1'b1;
            pc_fp0_load_vec  <= (~s_axis_pccmd_tdata[4]) ? s_axis_pccmd_tdata[10:6] : pc_fp0_load_vec;
            pc_fp1_load_vec <= s_axis_pccmd_tdata[4] ? s_axis_pccmd_tdata[10:6] : pc_fp1_load_vec;
        end else if (hs_axis_pccmd & (pccmd_type == 3'b100)) begin  // exec blk
            pc_exec_mode_sel <= s_axis_pccmd_tdata[4:3];
            psu_fpv_depth    <= s_axis_pccmd_tdata[14:6];
            psu_acc          <= s_axis_pccmd_tdata[15];
            psu_clr          <= s_axis_pccmd_tdata[16];
            execx_tile_sel   <= s_axis_pccmd_tdata[17];
            execy_tile_sel   <= s_axis_pccmd_tdata[18];
        end else if (hs_axis_pccmd & (pccmd_type == 3'b101)) begin  // exec fpv
            pc_exec_mode_sel         <= s_axis_pccmd_tdata[4:3];
            pc_fp_updt_en            <= s_axis_pccmd_tdata[5];
            pc_fp_updt_sel           <= s_axis_pccmd_tdata[6];
            pc_fp0_exec_vec          <= s_axis_pccmd_tdata[11:7];
            pc_fp1_exec_vec          <= s_axis_pccmd_tdata[16:12];
            pc_fp_updt_vec           <= s_axis_pccmd_tdata[21:17];
            const_fp16_reg_files_sel <= s_axis_pccmd_tdata[24:22];
            pc_fp_op_sel             <= s_axis_pccmd_tdata[27:25];
        end else if (hs_axis_pccmd & (pccmd_type == 3'b011)) begin  // store
            pc_store_mode_sel <= s_axis_pccmd_tdata[4:3];
            store_depth       <= s_axis_pccmd_tdata[12:5];
            store_tapu_depth  <= s_axis_pccmd_tdata[17:13];
            store_mt_en       <= s_axis_pccmd_tdata[18];
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
    assign done_state          = pc_loadx_done | pc_loady_done | pc_exec_done | pc_store_done;

    // logic pcfbk_en;
    // assign pcfbk_en = done_state & ((~m_axis_pcfbk_tvalid) | m_axis_pcfbk_tready);

    // logic hs_axis_pcfbk;
    // assign hs_axis_pcfbk = m_axis_pcfbk_tvalid & m_axis_pcfbk_tready;

    // logic pcfbk_valid_r;
    // always_ff @(posedge clk) begin
    //     if (~rst_n) begin
    //         pcfbk_valid_r <= 1'b0;
    //     end else if (pcfbk_en) begin
    //         pcfbk_valid_r <= 1'b1;
    //     end else if (hs_axis_pcfbk & (~pcfbk_en)) begin
    //         pcfbk_valid_r <= 1'b0;
    //     end
    // end
    // assign m_axis_pcfbk_tvalid = pcfbk_valid_r;
    assign m_axis_pcfbk_tvalid = done_state;
    assign m_axis_pcfbk_tdata  = {4'd0, pc_loady_done, pc_loadx_done, pc_exec_done, pc_store_done};

    // always_ff @(posedge clk) begin
    //     m_axis_pcfbk_tdata <= {4'd0, pc_loady_done, pc_loadx_done, pc_exec_done, pc_store_done};
    // end

    assign s_axis_pccmd_tready = 1'b1;

endmodule
