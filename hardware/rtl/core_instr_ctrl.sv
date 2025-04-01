`timescale 1ns / 1ps

/**************************************************** ISA Definition ****************************************************/

/**************************************************** ISA Definition ****************************************************/
module core_instr_ctrl #(
    parameter PCC_INSTR_WIDTH       = 64,
    parameter DATA_BASEADDR_WIDTH   = 40,
    parameter AXIS_PCCMD_DATA_WIDTH = 32,
    parameter AXIS_PCFBK_DATA_WIDTH = 8,
    parameter PSU_DEPTH_WIDTH       = 9,
    parameter BLK_NUM_WIDTH         = 5,
    parameter FPV_NUM_WIDTH         = 5,
    parameter QUANT_SCALE_EXP_WIDTH = 5,
    parameter QUANT_SCALE_MAN_WIDTH = 16
) (
    input  logic                             clk,
    input  logic                             rst_n,
    // debug status
    output logic [                     31:0] debug_status,
    output logic [                     31:0] latency_cycles,
    // base address
    input  logic [                     39:0] yizo_base_addr,
    input  logic [                     39:0] xi_base_addr,
    // input axis_instr
    output logic                             s_axis_instr_tready,
    input  logic                             s_axis_instr_tvalid,
    input  logic [      PCC_INSTR_WIDTH-1:0] s_axis_instr_tdata,
    // output stream - mm2s cmd
    output logic [                     87:0] m_axis_mm2s_cmd_xi_tdata,
    output logic                             m_axis_mm2s_cmd_xi_tvalid,
    input  logic                             m_axis_mm2s_cmd_xi_tready,
    output logic [                     87:0] m_axis_mm2s_cmd_yi_tdata,
    output logic                             m_axis_mm2s_cmd_yi_tvalid,
    input  logic                             m_axis_mm2s_cmd_yi_tready,
    // output stream - s2mm cmd
    output logic [                     87:0] m_axis_s2mm_cmd_zo_tdata,
    output logic                             m_axis_s2mm_cmd_zo_tvalid,
    input  logic                             m_axis_s2mm_cmd_zo_tready,
    // AXI stream (master) for micro instructions to processing kernel
    input  logic                             m_axis_pccmd_tready,
    output logic                             m_axis_pccmd_tvalid,
    output logic [AXIS_PCCMD_DATA_WIDTH-1:0] m_axis_pccmd_tdata,
    // AXI stream (slave) for feedback from processing kernel
    output logic                             s_axis_pcfbk_tready,
    input  logic                             s_axis_pcfbk_tvalid,
    input  logic [AXIS_PCFBK_DATA_WIDTH-1:0] s_axis_pcfbk_tdata
);

    /*********************************** Decode Instructions ***********************************/
    logic hs_axis_instr;
    logic instr_en;
    // one MATADD has been fused into two MATMUL ops
    // currently we do not support FPVDIV
    localparam LAYER_CONFIG = 3'b000;
    localparam PCLOADBLK = 3'b001;
    localparam PCLOADFPV = 3'b010;
    localparam PCEXECBLK = 3'b011;
    localparam PCEXECFPV = 3'b100;
    localparam PCSTORE = 3'b110;

    logic [2:0] instr_type_e;
    assign instr_type_e  = s_axis_instr_tdata[62:60];
    assign hs_axis_instr = s_axis_instr_tvalid & s_axis_instr_tready;
    assign instr_en      = hs_axis_instr & (s_axis_instr_tdata[63] == 1'b1);

    logic                             pc_load_mode_sel;
    logic [                      1:0] pc_exec_mode_sel;
    logic [                      1:0] pc_store_mode_sel;

    logic [      PSU_DEPTH_WIDTH-1:0] xi_load_depth;
    logic [                     15:0] xi_load_mm2s_btt;
    logic [  DATA_BASEADDR_WIDTH-1:0] xi_load_mm2s_addr;
    logic                             xi_load_tile_sel;
    logic [      PSU_DEPTH_WIDTH-1:0] yi_load_depth;
    logic [                     15:0] yi_load_mm2s_btt;
    logic [  DATA_BASEADDR_WIDTH-1:0] yi_load_mm2s_addr;
    logic                             yi_load_tile_sel;
    logic                             load_xi_or_yi;

    logic [      PSU_DEPTH_WIDTH-1:0] psu_fpv_depth;
    logic                             psu_acc;
    logic                             psu_clr;
    logic                             execx_tile_sel;
    logic                             execy_tile_sel;

    logic [                     15:0] zo_store_s2mm_btt;
    logic [  DATA_BASEADDR_WIDTH-1:0] zo_store_s2mm_addr;
    logic [                      7:0] store_depth;
    logic [                      4:0] store_tapu_depth;
    logic                             store_mt_en;

    logic [        FPV_NUM_WIDTH-1:0] pc_fp0_load_vec;
    logic [        FPV_NUM_WIDTH-1:0] pc_fp1_load_vec;
    logic                             pc_fp_load_sel;
    logic                             pc_fp_updt_en;
    logic                             pc_fp_updt_sel;
    logic [        FPV_NUM_WIDTH-1:0] pc_fp_updt_vec;
    logic [        FPV_NUM_WIDTH-1:0] pc_fp0_exec_vec;
    logic [        FPV_NUM_WIDTH-1:0] pc_fp1_exec_vec;
    logic [QUANT_SCALE_EXP_WIDTH-1:0] pc_quant_sf_exp;
    logic [QUANT_SCALE_MAN_WIDTH-1:0] pc_quant_sf_man;

    logic [                     15:0] config_const_fp16_reg;
    logic                             config_sf_or_const;
    logic [                      2:0] config_const_sel;
    logic [                      2:0] const_fp16_reg_files_sel;
    logic [                      2:0] pc_fp_op_sel;

    logic [                      7:0] store_depth_p1;
    assign store_depth_p1 = s_axis_instr_tdata[54:48] + 1;

    always_ff @(posedge clk) begin
        if (instr_en) begin
            if (instr_type_e == LAYER_CONFIG) begin
                if (s_axis_instr_tdata[59:58] == 2'b01) begin
                    config_sf_or_const <= 1'b1;
                    pc_quant_sf_exp    <= s_axis_instr_tdata[4:0];
                    pc_quant_sf_man    <= s_axis_instr_tdata[20:5];
                end else if (s_axis_instr_tdata[59:58] == 2'b00) begin
                    config_sf_or_const    <= 1'b0;
                    config_const_fp16_reg <= s_axis_instr_tdata[15:0];
                    config_const_sel      <= s_axis_instr_tdata[18:16];
                end
            end else if (instr_type_e == PCLOADBLK & (~s_axis_instr_tdata[57])) begin
                xi_load_mm2s_addr <= s_axis_instr_tdata[31:0] + xi_base_addr;
                xi_load_mm2s_btt  <= s_axis_instr_tdata[47:32];
                xi_load_depth     <= s_axis_instr_tdata[56:48];
                xi_load_tile_sel  <= s_axis_instr_tdata[58];
                load_xi_or_yi     <= 1'b0;
                pc_load_mode_sel  <= 1'b0;
            end else if (instr_type_e == PCLOADBLK & s_axis_instr_tdata[57]) begin
                yi_load_mm2s_addr <= s_axis_instr_tdata[31:0] + yizo_base_addr;
                yi_load_mm2s_btt  <= s_axis_instr_tdata[47:32];
                yi_load_depth     <= s_axis_instr_tdata[56:48];
                yi_load_tile_sel  <= s_axis_instr_tdata[58];
                load_xi_or_yi     <= 1'b1;
                pc_load_mode_sel  <= 1'b0;
            end else if (instr_type_e == PCLOADFPV & (~s_axis_instr_tdata[53])) begin
                xi_load_mm2s_addr <= s_axis_instr_tdata[31:0] + xi_base_addr;
                xi_load_mm2s_btt  <= s_axis_instr_tdata[47:32];
                pc_fp0_load_vec   <= s_axis_instr_tdata[52:48];
                pc_fp_load_sel    <= s_axis_instr_tdata[53];
                pc_load_mode_sel  <= 1'b1;
            end else if (instr_type_e == PCLOADFPV & s_axis_instr_tdata[53]) begin
                yi_load_mm2s_addr <= s_axis_instr_tdata[31:0] + yizo_base_addr;
                yi_load_mm2s_btt  <= s_axis_instr_tdata[47:32];
                pc_fp1_load_vec   <= s_axis_instr_tdata[52:48];
                pc_fp_load_sel    <= s_axis_instr_tdata[53];
                pc_load_mode_sel  <= 1'b1;
            end else if (instr_type_e == PCEXECBLK) begin
                pc_exec_mode_sel <= s_axis_instr_tdata[1:0];
                execx_tile_sel   <= s_axis_instr_tdata[2];
                execy_tile_sel   <= s_axis_instr_tdata[3];
                psu_fpv_depth    <= s_axis_instr_tdata[56:48];
                psu_acc          <= s_axis_instr_tdata[57];
                psu_clr          <= s_axis_instr_tdata[58];
            end else if (instr_type_e == PCEXECFPV) begin
                pc_exec_mode_sel         <= s_axis_instr_tdata[1:0];
                pc_fp0_exec_vec          <= s_axis_instr_tdata[36:32];
                pc_fp1_exec_vec          <= s_axis_instr_tdata[41:37];
                pc_fp_updt_en            <= s_axis_instr_tdata[42];
                pc_fp_updt_sel           <= s_axis_instr_tdata[43];
                pc_fp_updt_vec           <= s_axis_instr_tdata[48:44];
                const_fp16_reg_files_sel <= s_axis_instr_tdata[51:49];
                pc_fp_op_sel             <= s_axis_instr_tdata[55:53];
            end else if (instr_type_e == PCSTORE) begin
                zo_store_s2mm_addr <= s_axis_instr_tdata[31:0] + yizo_base_addr;
                zo_store_s2mm_btt  <= s_axis_instr_tdata[47:32];
                store_depth        <= s_axis_instr_tdata[55:48];
                store_tapu_depth   <= store_depth_p1[7:3] - 1;
                pc_store_mode_sel  <= s_axis_instr_tdata[58:57];
                store_mt_en        <= s_axis_instr_tdata[59];
            end
        end
    end
    /*********************************** Decode Instructions ***********************************/

    /************************************* Feedback from PC ************************************/
    logic pc_loadx_done, pc_loady_done, pc_exec_done, pc_store_done;
    logic hs_axis_pcfbk;
    assign hs_axis_pcfbk = s_axis_pcfbk_tvalid & s_axis_pcfbk_tready;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            pc_loady_done <= 1'b0;
            pc_loadx_done <= 1'b0;
            pc_exec_done  <= 1'b0;
            pc_store_done <= 1'b0;
        end else begin
            pc_loady_done <= hs_axis_pcfbk & s_axis_pcfbk_tdata[3];
            pc_loadx_done <= hs_axis_pcfbk & s_axis_pcfbk_tdata[2];
            pc_exec_done  <= hs_axis_pcfbk & s_axis_pcfbk_tdata[1];
            pc_store_done <= hs_axis_pcfbk & s_axis_pcfbk_tdata[0];
        end
    end
    assign s_axis_pcfbk_tready = 1'b1;
    /************************************* Feedback from PC ************************************/

    /**************************************** ILP Control ****************************************/
    logic loadx_busy, loady_busy, exec_busy, store_busy;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            loadx_busy <= 1'b0;
        end else if (instr_en & (((instr_type_e == PCLOADBLK) & (~s_axis_instr_tdata[57])) | ((instr_type_e == PCLOADFPV) & (~s_axis_instr_tdata[53])))) begin
            loadx_busy <= 1'b1;
        end else if (pc_loadx_done) begin
            loadx_busy <= 1'b0;
        end
    end
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            loady_busy <= 1'b0;
        end else if (instr_en & (((instr_type_e == PCLOADBLK) & s_axis_instr_tdata[57]) | ((instr_type_e == PCLOADFPV) & (s_axis_instr_tdata[53])))) begin
            loady_busy <= 1'b1;
        end else if (pc_loady_done) begin
            loady_busy <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            exec_busy <= 1'b0;
        end else if (instr_en & (instr_type_e == PCEXECBLK | instr_type_e == PCEXECFPV)) begin
            exec_busy <= 1'b1;
        end else if (pc_exec_done) begin
            exec_busy <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            store_busy <= 1'b0;
        end else if (instr_en & (instr_type_e == PCSTORE)) begin
            store_busy <= 1'b1;
        end else if (pc_store_done) begin
            store_busy <= 1'b0;
        end
    end

    // generate conflict signals
    logic ld_ld_conflict, ex_ex_conflict, st_st_conflict;
    logic ldx_ex_conflict, ldy_ex_conflict;
    logic ex_ld_conflict;
    logic ex_st_conflict;
    // logic layer_conflict;
    logic ilp_conflict;

    logic next_execx_tile_sel, next_execy_tile_sel, next_load_tile_sel;
    assign next_execx_tile_sel = s_axis_instr_tdata[2];
    assign next_execy_tile_sel = s_axis_instr_tdata[3];
    assign next_load_tile_sel = s_axis_instr_tdata[58];

    // @TODO: if the timing is not met, we can add a stall signal to stall the instruction
    assign ld_ld_conflict = ((((instr_type_e == PCLOADBLK) & (~s_axis_instr_tdata[57])) | ((instr_type_e == PCLOADFPV) & (~s_axis_instr_tdata[53]))) & (~m_axis_mm2s_cmd_xi_tready)) | ((((instr_type_e == PCLOADBLK) & s_axis_instr_tdata[57]) | ((instr_type_e == PCLOADFPV) & (s_axis_instr_tdata[53]))) & (~m_axis_mm2s_cmd_yi_tready));
    assign ex_ex_conflict = ((instr_type_e == PCEXECBLK)) & exec_busy;
    assign st_st_conflict = (instr_type_e == PCSTORE) & store_busy;
    assign ldx_ex_conflict = (instr_type_e == PCEXECBLK) & (next_execx_tile_sel == xi_load_tile_sel) & loadx_busy;
    assign ldy_ex_conflict = (instr_type_e == PCEXECBLK) & (next_execy_tile_sel == yi_load_tile_sel) & loady_busy;
    assign ex_ld_conflict = (instr_type_e == PCLOADBLK) & (((next_load_tile_sel == execx_tile_sel) & (~s_axis_instr_tdata[57])) | ((next_load_tile_sel == execy_tile_sel) & (s_axis_instr_tdata[57]))) & exec_busy;
    assign ex_st_conflict = (instr_type_e == PCSTORE) & exec_busy;
    // assign layer_conflict = (instr_type_e == LAYER_CONFIG) & store_busy;

    assign ilp_conflict = ld_ld_conflict | ex_ex_conflict | st_st_conflict | ldx_ex_conflict | ldy_ex_conflict | ex_ld_conflict | ex_st_conflict;
    assign s_axis_instr_tready = ~ilp_conflict;
    /**************************************** ILP Control ****************************************/

    /********************************** Generate Data Mover cmd ***********************************/
    // xy mm2s cmd
    logic [87:0] ld_xi_cmd;
    assign ld_xi_cmd[22:0]  = {7'd0, xi_load_mm2s_btt};  // BTT
    assign ld_xi_cmd[23]    = 1'b1;  // TYPE
    assign ld_xi_cmd[29:24] = 0;  // DSA
    assign ld_xi_cmd[30]    = 1'b1;  // EOF
    assign ld_xi_cmd[31]    = 1'b0;  // DRR
    assign ld_xi_cmd[71:32] = xi_load_mm2s_addr;  // SADDR
    assign ld_xi_cmd[87:72] = 0;

    logic [87:0] ld_yi_cmd;
    assign ld_yi_cmd[22:0]  = {7'd0, yi_load_mm2s_btt};  // BTT
    assign ld_yi_cmd[23]    = 1'b1;  // TYPE
    assign ld_yi_cmd[29:24] = 0;  // DSA
    assign ld_yi_cmd[30]    = 1'b1;  // EOF
    assign ld_yi_cmd[31]    = 1'b0;  // DRR
    assign ld_yi_cmd[71:32] = yi_load_mm2s_addr;  // SADDR
    assign ld_yi_cmd[87:72] = 0;

    // out cmd
    logic [87:0] st_zo_cmd;
    assign st_zo_cmd[22:0]          = {7'd0, zo_store_s2mm_btt};  // BTT
    assign st_zo_cmd[23]            = 1'b1;  // TYPE
    assign st_zo_cmd[29:24]         = 0;  // DSA
    assign st_zo_cmd[30]            = 1'b1;  // EOF
    assign st_zo_cmd[31]            = 1'b0;  // DRR
    assign st_zo_cmd[71:32]         = zo_store_s2mm_addr;  // SADDR
    assign st_zo_cmd[87:72]         = 0;

    // axi stream control
    assign m_axis_mm2s_cmd_xi_tdata = ld_xi_cmd;
    assign m_axis_mm2s_cmd_yi_tdata = ld_yi_cmd;

    logic xi_mover_valid_r;
    logic xi_mover_hs;
    logic xi_mover_en;
    logic ld_xi_cmd_en;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            ld_xi_cmd_en <= 1'b0;
        end else begin
            ld_xi_cmd_en <= instr_en & (((instr_type_e == PCLOADBLK) & (~s_axis_instr_tdata[57])) | ((instr_type_e == PCLOADFPV) & (~s_axis_instr_tdata[53])));
        end
    end

    logic yi_mover_valid_r;
    logic yi_mover_hs;
    logic yi_mover_en;
    logic ld_yi_cmd_en;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            ld_yi_cmd_en <= 1'b0;
        end else begin
            ld_yi_cmd_en <= instr_en & (((instr_type_e == PCLOADBLK) & s_axis_instr_tdata[57]) | ((instr_type_e == PCLOADFPV) & (s_axis_instr_tdata[53])));
        end
    end

    assign xi_mover_hs = m_axis_mm2s_cmd_xi_tready & m_axis_mm2s_cmd_xi_tvalid;
    assign xi_mover_en = ld_xi_cmd_en & ((~m_axis_mm2s_cmd_xi_tvalid) | m_axis_mm2s_cmd_xi_tready);

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            xi_mover_valid_r <= 1'b0;
        end else if (xi_mover_hs & (~xi_mover_en)) begin
            xi_mover_valid_r <= 1'b0;
        end else if (xi_mover_en) begin
            xi_mover_valid_r <= 1'b1;
        end
    end
    assign m_axis_mm2s_cmd_xi_tvalid = xi_mover_valid_r;

    assign yi_mover_hs = m_axis_mm2s_cmd_yi_tready & m_axis_mm2s_cmd_yi_tvalid;
    assign yi_mover_en = ld_yi_cmd_en & ((~m_axis_mm2s_cmd_yi_tvalid) | m_axis_mm2s_cmd_yi_tready);

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            yi_mover_valid_r <= 1'b0;
        end else if (yi_mover_hs & (~yi_mover_en)) begin
            yi_mover_valid_r <= 1'b0;
        end else if (yi_mover_en) begin
            yi_mover_valid_r <= 1'b1;
        end
    end
    assign m_axis_mm2s_cmd_yi_tvalid = yi_mover_valid_r;

    logic zo_mover_valid_r;
    logic zo_mover_hs;
    logic zo_mover_en;
    logic store_cmd_en;
    assign store_cmd_en = instr_en & (instr_type_e == PCSTORE);

    assign m_axis_s2mm_cmd_zo_tdata = st_zo_cmd;
    assign zo_mover_hs = m_axis_s2mm_cmd_zo_tready & m_axis_s2mm_cmd_zo_tvalid;
    assign zo_mover_en = (store_cmd_en) & ((~m_axis_s2mm_cmd_zo_tvalid)|m_axis_s2mm_cmd_zo_tready);

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            zo_mover_valid_r <= 1'b0;
        end else if (zo_mover_hs & (~zo_mover_en)) begin
            zo_mover_valid_r <= 1'b0;
        end else if (zo_mover_en) begin
            zo_mover_valid_r <= 1'b1;
        end
    end
    assign m_axis_s2mm_cmd_zo_tvalid = zo_mover_valid_r;
    /********************************** Generate Data Mover cmd ***********************************/

    /************************************** Generate PC cmd ***************************************/
    logic instr_gen_en;
    logic instr_gen_en_d;
    logic pcloadblk_gen_en;
    logic pcloadblk_gen_en_d;
    logic pcloadfpv_gen_en;
    logic pcloadfpv_gen_en_d;
    logic pcexecblk_gen_en;
    logic pcexecblk_gen_en_d;
    logic pcexecfpv_gen_en;
    logic pcexecfpv_gen_en_d;
    logic pcstore_gen_en;
    logic pcstore_gen_en_d;
    logic pcconfig_gen_en;
    logic pcconfig_gen_en_d;
    assign instr_gen_en     = instr_en;
    assign pcconfig_gen_en  = instr_en & (instr_type_e == LAYER_CONFIG);
    assign pcloadblk_gen_en = instr_en & (instr_type_e == PCLOADBLK);
    assign pcloadfpv_gen_en = instr_en & (instr_type_e == PCLOADFPV);
    assign pcexecblk_gen_en = instr_en & (instr_type_e == PCEXECBLK);
    assign pcexecfpv_gen_en = instr_en & (instr_type_e == PCEXECFPV);
    assign pcstore_gen_en   = instr_en & (instr_type_e == PCSTORE);

    always_ff @(posedge clk) begin
        instr_gen_en_d     <= instr_gen_en;
        pcloadblk_gen_en_d <= pcloadblk_gen_en;
        pcloadfpv_gen_en_d <= pcloadfpv_gen_en;
        pcexecblk_gen_en_d <= pcexecblk_gen_en;
        pcexecfpv_gen_en_d <= pcexecfpv_gen_en;
        pcstore_gen_en_d   <= pcstore_gen_en;
        pcconfig_gen_en_d  <= pcconfig_gen_en;
    end

    // logic pccmd_en;
    // assign pccmd_en = instr_gen_en_d & ((~m_axis_pccmd_tvalid) | m_axis_pccmd_tready);
    // always_ff @(posedge clk) begin
    //     if (~rst_n) begin
    //         m_axis_pccmd_tvalid <= 1'b0;
    //     end else if (m_axis_pccmd_tvalid & m_axis_pccmd_tready & (~pccmd_en)) begin
    //         m_axis_pccmd_tvalid <= 1'b0;
    //     end else if (pccmd_en) begin
    //         m_axis_pccmd_tvalid <= 1'b1;
    //     end
    // end
    assign m_axis_pccmd_tvalid = instr_gen_en_d;

    // // pc_exec_mode_sel is only updated in load commands
    // always_ff @(posedge clk) begin
    //     if (~rst_n) begin
    //         pc_load_mode_sel <= 0;
    //         load_tile_sel    <= 0;
    //         data_load_xy_sel <= 0;
    //         xi_load_depth    <= 0;
    //         yi_load_depth    <= 0;
    //         pc_fp_load_sel   <= 0;
    //         pc_fp_load_vec   <= 0;
    //         pc_exec_mode_sel      <= 0;
    //         exec_tile_sel    <= 0;
    //         psu_fpv_depth    <= 0;
    //         psu_acc          <= 0;
    //         pc_fp_updt_en    <= 0;
    //         pc_fp_updt_sel   <= 0;
    //         pc_fp0_exec_vec  <= 0;
    //         pc_fp1_exec_vec  <= 0;
    //         pc_fp_updt_vec   <= 0;
    //         pc_store_mode_sel<= 0;
    //         store_depth      <= 0;
    //         pc_quant_sf_exp  <= 0;
    //         pc_quant_sf_man  <= 0;
    //     end
    //     else if (hs_axis_pccmd & (pccmd_type == 3'b000)) begin // configuration
    //         pc_quant_sf_exp  <= s_axis_pccmd_tdata[7:3];
    //         pc_quant_sf_man  <= s_axis_pccmd_tdata[23:8];
    //     end
    //     else if (hs_axis_pccmd & (pccmd_type == 3'b001)) begin  // load int8
    //         pc_load_mode_sel <= s_axis_pccmd_tdata[3];
    //         load_tile_sel    <= s_axis_pccmd_tdata[4];
    //         xi_load_depth    <= s_axis_pccmd_tdata[14:6];
    //         yi_load_depth    <= s_axis_pccmd_tdata[10:6];
    //     end
    //     else if (hs_axis_pccmd & (pccmd_type == 3'b010)) begin // load fpv
    //         pc_load_mode_sel <= s_axis_pccmd_tdata[3];
    //         pc_fp_load_sel   <= s_axis_pccmd_tdata[4];
    //         data_load_xy_sel <= s_axis_pccmd_tdata[5];
    //         pc_fp_load_vec   <= s_axis_pccmd_tdata[10:6];
    //     end else if (hs_axis_pccmd & (pccmd_type == 3'b100)) begin  // exec blk
    //         pc_exec_mode_sel   <= {pc_load_mode_sel, s_axis_pccmd_tdata[3]};
    //         exec_tile_sel <= s_axis_pccmd_tdata[4];
    //         psu_fpv_depth <= s_axis_pccmd_tdata[14:6];
    //         psu_acc       <= s_axis_pccmd_tdata[15];
    //     end else if (hs_axis_pccmd & (pccmd_type == 3'b101)) begin  // exec fpv
    //         pc_exec_mode_sel     <= {pc_load_mode_sel, s_axis_pccmd_tdata[3]};
    //         pc_fp_updt_en   <= s_axis_pccmd_tdata[4];
    //         pc_fp_updt_sel  <= s_axis_pccmd_tdata[5];
    //         pc_fp0_exec_vec <= s_axis_pccmd_tdata[10:6];
    //         pc_fp1_exec_vec <= s_axis_pccmd_tdata[15:11];
    //         pc_fp_updt_vec  <= s_axis_pccmd_tdata[20:16];
    //     end else if (hs_axis_pccmd & (pccmd_type == 3'b011)) begin  // store
    //         pc_store_mode_sel <= s_axis_pccmd_tdata[3];
    //         store_depth       <= s_axis_pccmd_tdata[14:6];
    //         store_tapu_depth  <= s_axis_pccmd_tdata[17:15];
    //     end
    // end

    always_comb begin
        if (pcconfig_gen_en_d) begin
            m_axis_pccmd_tdata = config_sf_or_const ? {4'd0, pc_quant_sf_man, pc_quant_sf_exp, config_sf_or_const, config_const_sel, 3'b000} : {9'd0, config_const_fp16_reg, config_sf_or_const, config_const_sel, 3'b000};
        end else if (pcloadblk_gen_en_d) begin
            m_axis_pccmd_tdata = load_xi_or_yi ? {17'd0, yi_load_depth, 1'b0, yi_load_tile_sel, load_xi_or_yi, 3'b001} : {17'd0, xi_load_depth, 1'b0, xi_load_tile_sel, load_xi_or_yi, 3'b001};
        end else if (pcloadfpv_gen_en_d) begin
            m_axis_pccmd_tdata = pc_fp_load_sel ? {21'd0, pc_fp1_load_vec, 1'b1, pc_fp_load_sel, 4'b0010} : {21'd0, pc_fp0_load_vec, 1'b0, pc_fp_load_sel, 4'b0010};
        end else if (pcexecblk_gen_en_d) begin
            m_axis_pccmd_tdata = {
                13'd0,
                execy_tile_sel,
                execx_tile_sel,
                psu_clr,
                psu_acc,
                psu_fpv_depth,
                1'b0,
                pc_exec_mode_sel,
                3'b100
            };
        end else if (pcexecfpv_gen_en_d) begin
            m_axis_pccmd_tdata = {
                4'd0,
                pc_fp_op_sel,
                const_fp16_reg_files_sel,
                pc_fp_updt_vec,
                pc_fp1_exec_vec,
                pc_fp0_exec_vec,
                pc_fp_updt_sel,
                pc_fp_updt_en,
                pc_exec_mode_sel,
                3'b101
            };
        end else if (pcstore_gen_en_d) begin
            m_axis_pccmd_tdata = {
                13'd0, store_mt_en, store_tapu_depth, store_depth, pc_store_mode_sel, 3'b011
            };
        end else begin
            m_axis_pccmd_tdata = 0;
        end
    end

    /************************************** Generate PC cmd ***************************************/

    /******************************************* DEBUG *******************************************/
    logic latency_count_en;
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            latency_count_en <= 1'b0;
        end else if (instr_en & (~latency_count_en)) begin
            latency_count_en <= 1'b1;
        end else if (instr_en & (instr_type_e == LAYER_CONFIG) & (s_axis_instr_tdata[59])) begin
            latency_count_en <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            debug_status   <= 32'd0;
            latency_cycles <= 32'd0;
        end else if ((latency_count_en == 1'b0) & instr_en & (instr_type_e == LAYER_CONFIG)) begin
            latency_cycles <= 32'd0;
        end else begin
            debug_status[7:0] <= (instr_en == 1'b1) ? (debug_status[7:0] + 1) : debug_status[7:0];
            debug_status[15:8] <= pc_loady_done ? (debug_status[15:8] + 1) : debug_status[15:8];
            debug_status[23:16] <= pc_exec_done ? (debug_status[23:16] + 1) : debug_status[23:16];
            debug_status[31:24] <= pc_store_done ? (debug_status[31:24] + 1) : debug_status[31:24];
            latency_cycles <= (latency_count_en == 1'b1) ? (latency_cycles + 1) : latency_cycles;
        end
    end
    /******************************************* DEBUG *******************************************/

endmodule
