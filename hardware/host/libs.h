#include <iostream>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include <experimental/xrt_xclbin.h>
#include <experimental/xrt_ip.h>
#include <vector>
#include "ap_int.h"
#include "ap_fixed.h"

#include "axi_control_regs.h"
#include "mem_tag.h"

using std::ifstream;

