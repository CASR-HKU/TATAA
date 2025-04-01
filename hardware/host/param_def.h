#ifndef __PARAM_DEF_H__
#define __PARAM_DEF_H__

#define LAYER_NUM_t       1
#define HBM_CH_NUM        2
#define BLOCK_NUM         32
#define NUM_CU            15
#define NUM_HBM           HBM_CH_NUM * NUM_CU
#define FREQ              300000000
#define BFP_MULT_NUM_t    128
#define TAPU_NUM_t        2
#define MAC_OPS_t         2
#define XH_t              BLOCK_NUM * 8
#define XW_t              8
#define YW_t              8
#define X_SIZE            16384
#define Y_SIZE            16384
#define INSTR_NUM         228
#define INSTR_SIZE        INSTR_NUM * sizeof(uint64_t)

#endif