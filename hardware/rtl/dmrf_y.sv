`timescale 1ns / 1ps

/**************************************************** Data Layout Diagram ****************************************************/
// INT8 MODE:
//          256-bit                            TILE 0 (BRAM0)     8-bit               256-bit
//                          |--> PORT 0  | ******** ... ******** | --> --|
//                          |--> PORT 1  | ******** ... ******** | --> --|
//                          |--> PORT 2  | ******** ... ******** | --> --|  
//                          |            |          ...            ...   |
//                          |--> PORT 15 | ******** ... ******** | --> --| 
//                          |--> PORT 16 | ******** ... ******** | --> --|
//                          |--> PORT 17 | ******** ... ******** | --> --|  
//                          |--> PORT 18 | ******** ... ******** | --> --|   
//                          |                       ...            ...   |
//                          |--> PORT 31 | ******** ... ******** | --> --|      
// Sync loading 32 ports ==>|                  TILE 1 (BRAM1)            |==> To 1 TAPU (int8-opt in DSP)
//                          |--> PORT 0  | ######## ... ######## | --> --|
//                          |--> PORT 1  | ######## ... ######## | --> --|
//                          |--> PORT 2  | ######## ... ######## | --> --|  
//                          |            |          ...            ...   |
//                          |--> PORT 15 | ######## ... ######## | --> --| 
//                          |--> PORT 16 | ######## ... ######## | --> --| 
//                          |--> PORT 17 | ######## ... ######## | --> --|  
//                          |--> PORT 18 | ######## ... ######## | --> --|   
//                          |                       ...            ...   |
//                          |--> PORT 31 | ######## ... ######## | --> --|      
//                          ^                                            ^
//                   Load Tile Select                             Exec Tile Select

// BFLOAT16 MODE:
//          256-bit                                BRAM 0         8-bit            512-bit
//                          |--> PORT 0  | ******** ... ******** | --> 
//                          |--> PORT 1  | ******** ... ******** | --> 
//                          |--> PORT 2  | ******** ... ******** | -->   
//                          |            |          ...            ... 
//                          |--> PORT 15 | ******** ... ******** | -->  
//                          |--> PORT 16 | ******** ... ******** | --> 
//                          |--> PORT 17 | ******** ... ******** | -->   
//                          |--> PORT 18 | ******** ... ******** | -->    
//                          |                       ...            ... 
//                          |--> PORT 31 | ******** ... ******** | -->       
// Sync loading 32 ports ==>|                      BRAM 1              ==> To 1 TAPU (int8-opt in DSP)
//                          |--> PORT 0  | ######## ... ######## | --> 
//                          |--> PORT 1  | ######## ... ######## | --> 
//                          |--> PORT 2  | ######## ... ######## | -->   
//                          |            |          ...            ... 
//                          |--> PORT 15 | ######## ... ######## | -->  
//                          |--> PORT 16 | ######## ... ######## | -->  
//                          |--> PORT 17 | ######## ... ######## | -->   
//                          |--> PORT 18 | ######## ... ######## | -->    
//                          |                       ...            ... 
//                          |--> PORT 31 | ######## ... ######## | -->       
//                          ^        Double Buffer is integrated inside                 
//                      Load Select                             

// INT8 BUFFER CONFIGURATION:
// - Maximum int8 depth for one tile: 512

// BFLOAT16 BUFFER CONFIGURATION:
// - Fixed bfloat16 vector size: 16
// - Maximum bfloat16 vector num for each dmrfy: 16
// - Maximum bfloat16 processing depth: 32
/**************************************************** Data Layout Diagram ****************************************************/

module dmrf_y #(
    parameter AXIS_LOAD_DATA_WIDTH = 256,
    parameter SYS_DATA_WIDTH       = 32,
    parameter SYS_PORT_NUM         = 16,
    parameter BRAM_SIZE            = 8192,
    parameter BRAM_ADDR_WIDTH      = 5,
    parameter BRAM_DATA_WIDTH      = 256,
    parameter LOAD_ADDR_WIDTH      = 5,
    parameter EXEC_ADDR_WIDTH      = 5,
    parameter EXP_BUF_DATA_WIDTH   = 32
) (
    // common signals
    input  logic                                   clk,
    input  logic                                   rst_n,
    // input axis for load
    input  logic [       AXIS_LOAD_DATA_WIDTH-1:0] s_axis_dmrfy0_load_tdata,
    input  logic                                   s_axis_dmrfy0_load_tvalid,
    output logic                                   s_axis_dmrfy0_load_tready,
    input  logic [       AXIS_LOAD_DATA_WIDTH-1:0] s_axis_dmrfy1_load_tdata,
    input  logic                                   s_axis_dmrfy1_load_tvalid,
    output logic                                   s_axis_dmrfy1_load_tready,
    // control
    input  logic [            LOAD_ADDR_WIDTH-1:0] dmrfy_load_depth,
    input  logic                                   dmrfy_mode_sel,
    input  logic [            EXEC_ADDR_WIDTH-1:0] dmrfy_exec_addr,
    output logic                                   dmrfy0_load_done,
    output logic                                   dmrfy1_load_done,
    // fp data
    input  logic [       AXIS_LOAD_DATA_WIDTH-1:0] dmrfy_fp_updt_data,
    input  logic                                   dmrfy_fp_updt_en,
    input  logic                                   dmrfy_fp_updt_sel,
    input  logic [            LOAD_ADDR_WIDTH-1:0] dmrfy_fp0_updt_addr,
    input  logic [            LOAD_ADDR_WIDTH-1:0] dmrfy_fp0_exec_addr,
    input  logic [            LOAD_ADDR_WIDTH-1:0] dmrfy_fp1_updt_addr,
    input  logic [            LOAD_ADDR_WIDTH-1:0] dmrfy_fp1_exec_addr,
    // output data
    output logic [SYS_DATA_WIDTH*SYS_PORT_NUM-1:0] dmrfy_exec_data
);

    // AXIS handshake
    logic hs_axis_dmrfy0_load, hs_axis_dmrfy1_load;
    assign hs_axis_dmrfy0_load = s_axis_dmrfy0_load_tvalid & s_axis_dmrfy0_load_tready;
    assign hs_axis_dmrfy1_load = s_axis_dmrfy1_load_tvalid & s_axis_dmrfy1_load_tready;

    // BRAM
    logic [     BRAM_ADDR_WIDTH-1:0] buf_load_addr0;
    logic [     BRAM_ADDR_WIDTH-1:0] buf_exec_addr0;
    logic [     BRAM_ADDR_WIDTH-1:0] buf_load_addr1;
    logic [     BRAM_ADDR_WIDTH-1:0] buf_exec_addr1;
    logic [AXIS_LOAD_DATA_WIDTH-1:0] buf_dmrfy0_ld_data;
    logic [AXIS_LOAD_DATA_WIDTH-1:0] buf_dmrfy1_ld_data;
    logic [     BRAM_DATA_WIDTH-1:0] dmrfy0_output_data;
    logic [     BRAM_DATA_WIDTH-1:0] dmrfy1_output_data;
    logic                            dmrfy0_ld_wr_en;
    logic                            dmrfy1_ld_wr_en;

    bram_sdp_wrapper #(
        .BUF_LD_ADDR_WIDTH(BRAM_ADDR_WIDTH),
        .BUF_LD_DATA_WIDTH(BRAM_DATA_WIDTH),
        .BUF_EX_ADDR_WIDTH(BRAM_ADDR_WIDTH),
        .BUF_EX_DATA_WIDTH(BRAM_DATA_WIDTH),
        .BUF_MEM_SIZE     (BRAM_SIZE),
        .MEMORY_PRIMITIVE ("block")
    ) bram_dmrfy0_inst (
        .clk         (clk),
        .buf_ld_wr_en(dmrfy0_ld_wr_en),
        .buf_ld_addr (buf_load_addr0),
        .buf_ld_data (buf_dmrfy0_ld_data),
        .buf_ex_addr (buf_exec_addr0),
        .buf_ex_data (dmrfy0_output_data)
    );

    always_ff @(posedge clk) begin
        dmrfy0_ld_wr_en <= (hs_axis_dmrfy0_load) | (dmrfy_fp_updt_en & (~dmrfy_fp_updt_sel));
        buf_dmrfy0_ld_data <= dmrfy_fp_updt_en ? dmrfy_fp_updt_data : s_axis_dmrfy0_load_tdata;
    end

    bram_sdp_wrapper #(
        .BUF_LD_ADDR_WIDTH(BRAM_ADDR_WIDTH),
        .BUF_LD_DATA_WIDTH(BRAM_DATA_WIDTH),
        .BUF_EX_ADDR_WIDTH(BRAM_ADDR_WIDTH),
        .BUF_EX_DATA_WIDTH(BRAM_DATA_WIDTH),
        .BUF_MEM_SIZE     (BRAM_SIZE),
        .MEMORY_PRIMITIVE ("block")
    ) bram_dmrfy1_inst (
        .clk         (clk),
        .buf_ld_wr_en(dmrfy1_ld_wr_en),
        .buf_ld_addr (buf_load_addr1),
        .buf_ld_data (buf_dmrfy1_ld_data),
        .buf_ex_addr (buf_exec_addr1),
        .buf_ex_data (dmrfy1_output_data)
    );

    always_ff @(posedge clk) begin
        dmrfy1_ld_wr_en <= (hs_axis_dmrfy1_load) | (dmrfy_fp_updt_en & dmrfy_fp_updt_sel);
        buf_dmrfy1_ld_data <= dmrfy_fp_updt_en ? dmrfy_fp_updt_data : s_axis_dmrfy1_load_tdata;
    end

    // select exec data, crossing two BRAMs
    // @TODO: parameterized
    assign dmrfy_exec_data = {
        dmrfy1_output_data[255:240],
        dmrfy0_output_data[255:240],
        dmrfy1_output_data[239:224],
        dmrfy0_output_data[239:224],
        dmrfy1_output_data[223:208],
        dmrfy0_output_data[223:208],
        dmrfy1_output_data[207:192],
        dmrfy0_output_data[207:192],
        dmrfy1_output_data[191:176],
        dmrfy0_output_data[191:176],
        dmrfy1_output_data[175:160],
        dmrfy0_output_data[175:160],
        dmrfy1_output_data[159:144],
        dmrfy0_output_data[159:144],
        dmrfy1_output_data[143:128],
        dmrfy0_output_data[143:128],
        dmrfy1_output_data[127:112],
        dmrfy0_output_data[127:112],
        dmrfy1_output_data[111:96],
        dmrfy0_output_data[111:96],
        dmrfy1_output_data[95:80],
        dmrfy0_output_data[95:80],
        dmrfy1_output_data[79:64],
        dmrfy0_output_data[79:64],
        dmrfy1_output_data[63:48],
        dmrfy0_output_data[63:48],
        dmrfy1_output_data[47:32],
        dmrfy0_output_data[47:32],
        dmrfy1_output_data[31:16],
        dmrfy0_output_data[31:16],
        dmrfy1_output_data[15:0],
        dmrfy0_output_data[15:0]
    };

    // load control
    logic [LOAD_ADDR_WIDTH-1:0] cnt_dmrfy_load;
    logic                       full_cnt_dmrfy_load;
    logic                       hs_axis_dmrfy_load;

    assign hs_axis_dmrfy_load = hs_axis_dmrfy0_load | hs_axis_dmrfy1_load;

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            cnt_dmrfy_load <= 0;
        end else if (hs_axis_dmrfy_load) begin
            if (cnt_dmrfy_load == dmrfy_load_depth) begin
                cnt_dmrfy_load <= 0;
            end else begin
                cnt_dmrfy_load <= cnt_dmrfy_load + 1;
            end
        end
    end
    assign full_cnt_dmrfy_load = hs_axis_dmrfy_load & (cnt_dmrfy_load == dmrfy_load_depth);

    // @IMPORTANT: in bfp8 mode, double buffer is using two BRAMs; in bfloat16 mode, double buffer is integrated inside one BRAM
    // Therefore, the load & exec address should be different
    always_ff @(posedge clk) begin
        buf_load_addr0 <= dmrfy_mode_sel ? dmrfy_fp0_updt_addr : cnt_dmrfy_load;
        buf_exec_addr0 <= dmrfy_mode_sel ? dmrfy_fp0_exec_addr : dmrfy_exec_addr;
        buf_load_addr1 <= dmrfy_mode_sel ? dmrfy_fp1_updt_addr : cnt_dmrfy_load;
        buf_exec_addr1 <= dmrfy_mode_sel ? dmrfy_fp1_exec_addr : dmrfy_exec_addr;
    end
    // assign buf_load_addr0           = dmrfy_mode_sel ? dmrfy_fp0_updt_addr : cnt_dmrfy_load;
    // assign buf_exec_addr0           = dmrfy_mode_sel ? dmrfy_fp0_exec_addr : dmrfy_exec_addr;
    // assign buf_load_addr1           = dmrfy_mode_sel ? dmrfy_fp1_updt_addr : cnt_dmrfy_load;
    // assign buf_exec_addr1           = dmrfy_mode_sel ? dmrfy_fp1_exec_addr : dmrfy_exec_addr;

    // interface control
    assign s_axis_dmrfy0_load_tready = 1'b1;
    assign s_axis_dmrfy1_load_tready = 1'b1;
    assign dmrfy0_load_done          = dmrfy_mode_sel ? hs_axis_dmrfy0_load : full_cnt_dmrfy_load;
    assign dmrfy1_load_done          = dmrfy_mode_sel ? hs_axis_dmrfy1_load : full_cnt_dmrfy_load;

endmodule
