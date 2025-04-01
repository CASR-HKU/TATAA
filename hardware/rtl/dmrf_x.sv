`timescale 1ns / 1ps

/**************************************************** Data Layout Diagram ****************************************************/
//          256-bit                             TILE 1                  TILE 0         8-bit               MUX     128-bit
//                            PORT 0  | ######## ... ######## | ******** ... ******** | -->
//                            PORT 1  | ######## ... ######## | ******** ... ******** | -->
//                            PORT 2  | ######## ... ######## | ******** ... ******** | -->   STREAM 0 ==> ---
//                                               ...         ...         ...            ...                  |
//                            PORT 15 | ######## ... ######## | ******** ... ******** | -->                  |
// Sync loading 32 ports ==>  -------------------------------------------------------------------------      | ==> To 4 TAPUs
//                            PORT 16 | @@@@@@@@ ... @@@@@@@@ | %%%%%%%% ... %%%%%%%% | -->                  |
//                            PORT 17 | @@@@@@@@ ... @@@@@@@@ | %%%%%%%% ... %%%%%%%% | -->                  |
//                            PORT 18 | @@@@@@@@ ... @@@@@@@@ | %%%%%%%% ... %%%%%%%% | -->   STREAM 1 ==> ---
//                                               ...         ...         ...            ...                  ^
//                            PORT 31 | @@@@@@@@ ... @@@@@@@@ | %%%%%%%% ... %%%%%%%% | -->              stream_sel
//                                           DEPTH = 128             BLOCKS = 32
//                                                   Shared Address Space

// MICRO INSTRUCTIONS FOR MATMUL: (using depth as the unit)
// - LOAD TILE0 128
// - EXEC TILE0 STREAM0 128 
// - EXEC TILE0 STREAM1 128
// - LOAD TILE1 128         : this instuction can be executed in parallel due to double buffering
// - EXEC TILE1 STREAM0 128
// - EXEC TILE1 STREAM1 128
// - ...

// CONFIGURATION:
// - Maximum depth for one stream: 512
/**************************************************** Data Layout Diagram ****************************************************/

module dmrf_x #(
    parameter AXIS_LOAD_DATA_WIDTH = 256,
    parameter TAPU_NUM             = 8,
    parameter SYS_DATA_WIDTH       = 8,
    parameter SYS_PORT_NUM         = 32,
    parameter BRAM_SIZE            = 262144,
    parameter BRAM_ADDR_WIDTH      = 10,
    parameter BRAM_DATA_WIDTH      = 256,
    parameter LOAD_ADDR_WIDTH      = 9,
    parameter EXEC_ADDR_WIDTH      = 9
) (
    // common signals
    input  logic                                   clk,
    input  logic                                   rst_n,
    // input axis for load
    input  logic [       AXIS_LOAD_DATA_WIDTH-1:0] s_axis_dmrfx_load_tdata,
    input  logic                                   s_axis_dmrfx_load_tvalid,
    output logic                                   s_axis_dmrfx_load_tready,
    input  logic [     AXIS_LOAD_DATA_WIDTH/8-1:0] s_axis_dmrfx_load_tkeep,
    input  logic                                   s_axis_dmrfx_load_tlast,
    // control
    input  logic                                   dmrfx_load_tile_sel,
    input  logic [            EXEC_ADDR_WIDTH-1:0] dmrfx_exec_addr,
    input  logic                                   dmrfx_exec_tile_sel,
    output logic                                   dmrfx_load_done,
    input  logic [            LOAD_ADDR_WIDTH-1:0] dmrfx_load_depth,
    // output data
    output logic [SYS_DATA_WIDTH*SYS_PORT_NUM-1:0] dmrfx_exec_data
);

    // AXIS handshake
    logic hs_axis_dmrfx_load;
    assign hs_axis_dmrfx_load = s_axis_dmrfx_load_tvalid & s_axis_dmrfx_load_tready;

    // BRAM
    logic [LOAD_ADDR_WIDTH-1:0] dmrfx_load_addr;
    logic [BRAM_DATA_WIDTH-1:0] dmrfx_output_data;

    bram_sdp_wrapper #(
        .BUF_LD_ADDR_WIDTH(BRAM_ADDR_WIDTH),
        .BUF_LD_DATA_WIDTH(BRAM_DATA_WIDTH),
        .BUF_EX_ADDR_WIDTH(BRAM_ADDR_WIDTH),
        .BUF_EX_DATA_WIDTH(BRAM_DATA_WIDTH),
        .BUF_MEM_SIZE     (BRAM_SIZE)
    ) bram_dmrfx_inst (
        .clk         (clk),
        .buf_ld_wr_en(hs_axis_dmrfx_load),
        .buf_ld_addr ({dmrfx_load_tile_sel, dmrfx_load_addr}),
        .buf_ld_data (s_axis_dmrfx_load_tdata),
        .buf_ex_addr ({dmrfx_exec_tile_sel, dmrfx_exec_addr}),
        .buf_ex_data (dmrfx_output_data)
    );

    // select exec data
    assign dmrfx_exec_data = dmrfx_output_data;

    // load control
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            dmrfx_load_addr <= 0;
        end else if (hs_axis_dmrfx_load) begin
            if (dmrfx_load_addr == dmrfx_load_depth) begin
                dmrfx_load_addr <= 0;
            end else begin
                dmrfx_load_addr <= dmrfx_load_addr + 1;
            end
        end
    end

    // interface control
    assign s_axis_dmrfx_load_tready = 1'b1;
    assign dmrfx_load_done          = hs_axis_dmrfx_load & s_axis_dmrfx_load_tlast; 

endmodule
