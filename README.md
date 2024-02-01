# OSML
OSML is an ML-based scheduler that intelligently schedules multiple interactive resources to meet co-located services' QoS targets. OSML employs multiple ML models to work collaboratively to predict QoS variations, shepherd the scheduling, and recover from QoS violations in complicated co-location cases.

OSML can generalize across platforms. We provide OSML's implementation, data set, and ML models on a server equipped with an Intel Xeon E5-2697 v4 CPU ([Artifact link](https://github.com/Sys-Inventor-Lab/AI4System-OSML/blob/master/server_with_Intel_Xeon_E5_2697_v4)). We will release OSML on other platforms in the near future.

## Artifact list

1. OSML on a server equipped with an Intel Xeon E5-2697 v4 CPU
    - Link: [server_with_Intel_Xeon_E5_2697_v4](https://github.com/Sys-Inventor-Lab/AI4System-OSML/blob/master/server_with_Intel_Xeon_E5_2697_v4)
    - Platform configuration:
        | Configuration           | Specification           |
        | :---------------------: | :---------------------: | 
        | CPU Model               | Intel Xeon E5-2697 v4   |
        | Logical Processor Cores | 36 Cores (18 phy.cores) |
        | Processor Speed         | 2.3GHz                  |
        | Main Memory / Channel / BW | 256GB, 2400MHz DDR4 / 4 Channels / 76.8GB/s|
        | Private L1 & L2 Cache Size | 32KB and 256KB |
        | Shared L3 Cache Size | 45MB - 20 ways |
        | 45MB - 20 ways | 1TB, 7200 RPM, HD |
        |GPU |NVIDIA GP104 [GTX 1080], 8GB Memory|


