Link of OSML+ paper [Is Intelligence the Right Direction in New OS Scheduling for Multiple Resources in Cloud Environments?](https://dl.acm.org/doi/epdf/10.1145/3736584)

Link of OSML paper: [Intelligent Resource Scheduling for Co-located Latency-critical Services: A Multi-Model Collaborative Learning Approach](https://www.usenix.org/system/files/fast23-liu.pdf)

OSML and OSML+ were primarily developed by Lei Liu (PI) and Xinglei Dou.

# [29/Aug/2024] OSML+
OSML+ is the extension of OSML. The additional features of OSML+ over OSML include: (1) OSML+ has new ML models with a shorter scheduling time interval, making OSML+ converge faster. (2) We use transfer learning technology to make OSML+ work well on new platforms. (3) We design a new central control framework in OSML+ that handles complicated co-location cases including LC and BE services. 

OSML+ can generalize across platforms. We provide OSML+'s implementation, data set, and ML models on the following platforms.

## Platform list
1. **OSML+ on a server equipped with an Intel Xeon E5-2697 v4 CPU**
    - We have collected extensive traces on this platform. ML models can be trained and generalized based on these data and then used on new platforms with low-overhead transfer learning. The ML models on this platform were trained directly using these traces without transfer learning.
    - Link: [OSML+_on_server_with_Intel_Xeon_E5_2697_v4](https://github.com/Sys-Inventor-Lab/AI4System-OSML/blob/master/OSML+_on_server_with_Intel_Xeon_E5_2697_v4)
    - Platform configuration:
        | Configuration | Specification |
        | :---------------------: | :---------------------: | 
        | CPU Model | Intel Xeon E5-2697 v4 |
        | Logical Processor Cores | 36 Cores (18 physical cores) |
        | Processor Speed | 2.3 GHz |
        | Main Memory / Channel per-socket / BW per-socket | 256 GB, 2400 MHz DDR4 / 4 Channels / 76.8 GB/s|
        | L1I, L1D & L2 Cache Size | 32 KB, 32 KB and 256 KB |
        | Shared L3 Cache Size | 45 MB - 20 ways |
        | Disk | 1 TB, 7200 RPM, HD |
        | GPU | NVIDIA GP104 [GTX 1080], 8 GB Memory |


2. **OSML+ on a server equipped with an Intel Xeon Gold 6338 CPU**
    - OSML+ can be generalized to new platforms using transfer learning. We collect traces for only a few hours on this platform. The ML models are trained using transfer learning based on pre-trained models from [the server with Intel Xeon E5 2697 v4](https://github.com/Sys-Inventor-Lab/AI4System-OSML/blob/master/OSML+_on_server_with_Intel_Xeon_E5_2697_v4). The ML models perform well on the new platform.
    - Link: [OSML+_on_server_with_Intel_Xeon_Gold_6338](https://github.com/Sys-Inventor-Lab/AI4System-OSML/blob/master/OSML+_on_server_with_Intel_Xeon_Gold_6338)
    - Platform configuration:
        | Configuration           | Specification           |
        | :---------------------: | :---------------------: | 
        | CPU Model               | Intel Xeon Gold 6338    |
        | Logical Processor Cores | 64 Cores (32 physical cores) |
        | Processor Speed         | 2.0 GHz                  |
        | Main Memory / Channel per-socket / BW per-socket | 256 GB, 2933MHz DDR4 / 4 Channels / 94.0GB/s|
        | L1I, L1D & L2 Cache Size | 32 KB, 48 KB and 1.25 MB |
        | Shared L3 Cache Size | 48 MB - 12 ways |
        | Disk | 2 TB, SSD |
        | GPU | NVIDIA GeForce RTX 3080 LHR |


3. **OSML+ on a server equipped with an Intel Xeon Gold 5220R CPU**
    - On this platform, we also collect traces for a few hours and train the models using transfer learning based on pre-trained models from [the server with Intel Xeon E5 2697 v4](https://github.com/Sys-Inventor-Lab/AI4System-OSML/blob/master/OSML+_on_server_with_Intel_Xeon_E5_2697_v4). The ML models perform well on the new platform.
    - Link: [OSML+_on_server_with_Intel_Xeon_5220R](https://github.com/Sys-Inventor-Lab/AI4System-OSML/blob/master/OSML+_on_server_with_Intel_Xeon_Gold_5220R)
    - Platform configuration:
        | Configuration           | Specification           |
        | :---------------------: | :---------------------: | 
        | CPU Model               | Intel Xeon Gold 5220R   |
        | Logical Processor Cores | 48 Cores (24 physical cores) |
        | Processor Speed         | 2.2 GHz                  |
        | Main Memory / Channel per-socket / BW per-socket | 128 GB, 3200MHz DDR4 / 4 Channels / 102.4 GB/s|
        | L1I, L1D & L2 Cache Size | 32 KB, 32 KB and 1 MB |
        | Shared L3 Cache Size | 35.75 MB - 11 ways |
        | Disk | 500 GB, SSD |
        | GPU | NVIDIA GP104 [GTX 1080], 8 GB Memory |

<br/>
<br/>
  
# [29/Jan/2024] OSML
OSML is an ML-based scheduler that intelligently schedules multiple interactive resources to meet co-located services' QoS targets. OSML employs multiple ML models to work collaboratively to predict QoS variations, shepherd the scheduling, and recover from QoS violations in complicated co-location cases.

OSML can generalize across platforms. We provide OSML's implementation, data set, and ML models on a server equipped with an Intel Xeon E5-2697 v4 CPU ([link](https://github.com/Sys-Inventor-Lab/AI4System-OSML/blob/master/OSML_on_server_with_Intel_Xeon_E5_2697_v4)). We will release OSML on other platforms in the near future.

## Platform list
1. **OSML on a server equipped with an Intel Xeon E5-2697 v4 CPU**
    - Link: [server_with_Intel_Xeon_E5_2697_v4](https://github.com/Sys-Inventor-Lab/AI4System-OSML/blob/master/OSML_on_server_with_Intel_Xeon_E5_2697_v4)
    - Platform configuration:
        | Configuration           | Specification           |
        | :---------------------: | :---------------------: | 
        | CPU Model               | Intel Xeon E5-2697 v4   |
        | Logical Processor Cores | 36 Cores (18 phy.cores) |
        | Processor Speed         | 2.3 GHz                  |
        | Main Memory / Channel per-socket / BW per-socket | 256 GB, 2400 MHz DDR4 / 4 Channels / 76.8 GB/s|
        | L1I, L1D & L2 Cache Size | 32 KB, 32 KB and 256 KB |
        | Shared L3 Cache Size | 45 MB - 20 ways |
        | Disk | 1 TB, 7200 RPM, HD |
        | GPU | NVIDIA GP104 [GTX 1080], 8 GB Memory |
