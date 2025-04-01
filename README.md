# TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture

The open-source implementation of the paper "TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture" in ACM Transactions on Reconfigurable Technology and Systems.

## Compilation

The compiler of TATAA to parse Transformer models, generate dataflow and instructions for TATAA processor.

Please refer to the `./compilation` directory for more details.

## Hardware 

The hardware implementation of TATAA processor.

Please refer to the `./hardware` directory for more details.

## Quantization

The quantization tool to quantize Transformer models in TATAA (int8 + bfloat16).

Please refer to the `./quantization` directory for more details.

Also, we listed the required python environment in `./quantization`.

## Citation (ACM Format)
```
@article{TATAA,
author = {Wu, Jiajun and Song, Mo and Zhao, Jingmin and Gao, Yizhao and Li, Jia and So, Hayden Kwok-Hay},
title = {TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture},
year = {2025},
issue_date = {March 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {18},
number = {1},
issn = {1936-7406},
url = {https://doi-org.eproxy.lib.hku.hk/10.1145/3714416},
doi = {10.1145/3714416},
journal = {ACM Trans. Reconfigurable Technol. Syst.},
month = mar,
articleno = {14},
numpages = {31},
keywords = {Transformer acceleration, mixed integer-floating-point inference, transformable arithmetic architecture, non-linear arithmetic operations, systolic array, SIMD, FPGA}
}
```