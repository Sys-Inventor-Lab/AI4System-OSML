# OSML+
OSML+ is a new ML-based resource scheduling mechanism for co-located cloud services. It intelligently schedules multiple interactive resources to meet co-located services' QoS targets. OSML+ uses a multi-model collaborative learning approach during its scheduling and thus can handle complicated cases, e.g., avoiding resource cliffs, sharing resources among applications, enabling different scheduling policies for applications with different priorities, etc. OSML+ can generalize across platforms.

## Platform specification
This artifact is conducted on a server equipped with Intel Xeon E5-2697 v4. The platform specification is as following.

###### Table 1. Platform specification
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


## Pre-requirements and Benchmark Setup
1. Install Docker following instructions at [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/).

2. Install python dependencies.
```bash
python3 -m pip install -r requirements.txt
```

3. Pull the image containing benchmark applications from Docker Hub.
``` bash
docker pull sysinventor/osml_benchmark:v1.0
```

4. Prepare input data for benchmark applications following instructions in [apps/README.md](https://github.com/Sys-Inventor-Lab/AI4System-OSML/tree/master/OSML+_on_server_with_Intel_Xeon_E5_2697_v4/apps/README.md). We use volumes so that the applications in the docker container can read the input data from the host machine.

5. Install Intel's [pqos tool](https://github.com/intel/intel-cmt-cat).
```bash
wget https://github.com/intel/intel-cmt-cat/archive/refs/tags/v24.05.tar.gz
tar -xvf intel-cmt-cat-24.05.tar.gz
cd intel-cmt-cat-24.05/
make -j 32
make install
cd ../
```

6. Please note that some items in `configs.py` may need to be configured based on the server you are using, such as `MAX_LOAD`, `QOS_TARGET` and `RPS_COLLECTED`, which may vary on different servers.

## How to Run OSML+
Run the following instruction.
```bash
python3 osml_plus.py
```

The OSML+ scheduler automatically launches the applications in the docker container according to `workload.txt` and schedules them. You can edit `workload.txt` to configure the applications to be launched. An example is given below. This example means that the scheduler will launch 3 applications including sphinx, img-dnn, specjbb, each with 30%, 30% and 90% of its maxload. Each application starts 48 (the number of logical cores on the platform, i.e., `N_CORES` in `configs.py`) threads. The RPS of sphinx is raised to 50% of its maxload at 20 seconds.
```
# 1. Add an LC application to the workload (one line for each application, split parameters with blanks)
# Parameters:
#   - Name of the application
#   - Parse the next parameter as RPS or the percentage of the max load (Available options: ["RPS", "PCT"])
#   - RPS value or percentage value
#   - [Optional] Number of threads (default as N_CORES in configs.py)
#   - [Optional] Launch time (The unit is seconds, default as 0)
#   - [Optional] End time (The unit is seconds, default as None)
# Example:
sphinx PCT 30 48
img-dnn PCT 30 48
specjbb PCT 90 48

# 2. Change the RPS of an LC application (only supported for Tailbench applications, one line for each RPS changing request, split parameters with blanks)
# Parameters:
#   - "CHANGE_RPS"
#   - Name
#   - Parse the next parameter as RPS or the percentage of the max load (Available options: ["RPS", "PCT"])
#   - RPS value or percentage value
#   - Time point when changing the RPS (The unit is seconds)
# Example:
CHANGE_RPS sphinx PCT 50 20

# 3. Add a BE service to the workload (one line for each BE service, split parameters with blanks)
# Parameters:
#   - "BE"
#   - Name
# Example:
BE blackscholes
```

During the scheduling process, the OSML+ scheduler records the response latency and resource allocation of each application, and finally outputs them to the `logs/` directory.

## How to Obtain OSML+ Dataset
We have collected extensive real traces for widely deployed LC services on a platform equipped with an Intel Xeon E5-2697 v4 @ 2.3GHz CPU and 256GB memory. The traces are in a docker image. You can obtain the dataset using the following instructions.
```
docker pull sysinventor/osml_plus_dataset_E5_2697_v4:v1.0
docker run -idt --name osml_plus_dataset_E5_2697_v4 sysinventor/osml_plus_dataset_E5_2697_v4:v1.0 /bin/bash
docker cp osml_plus_dataset_E5_2697_v4:/home/data .
```

## How to Collect Your Own Dataset
1. Run `collect_multiple.py` in the `data/scripts/` directory to collect traces with background applications.
   
2. Run `generate_dataset.sh` in the `data/scripts/` directory to label the raw data, and generated the dataset used for ML model training.

Data in the `data_collection` directory are raw data. Data in the `data_process` directory are processed dataset used for model training. Run `python count_samples.py` to see how many samples are covered by the dataset.

## How to Train and Test the ML Models
1. The data set collected on this platform has a rich set of information. ML models can be trained and generalized based on these data and then used on new platforms with low-overhead transfer learning. In the `[models/](https://github.com/Sys-Inventor-Lab/AI4System-OSML/tree/master/OSML+_on_server_with_Intel_Xeon_E5_2697_v4/models/)` folder, `Model_A.py` and `Model_B.py` are the scripts for training the ML models. The ML models can be trained using the following instructions.
```
cd models/

# Train models using data set collected on the this platform. Transfer learning is not enabled.
python Model_A.py
python Model_B.py
```

2. The ML models trained on this platform are used as pre-trained models to generalize OSML+ on new platforms using transfer learning. We provide transfer learning examples on two new platforms ([OSML+_on_server_with_Intel_Xeon_Gold_5220R](https://github.com/Sys-Inventor-Lab/AI4System-OSML/tree/master/OSML+_on_server_with_Intel_Xeon_Gold_5220R/) and [server_with_Intel_Xeon_Gold_6338](https://github.com/Sys-Inventor-Lab/AI4System-OSML/tree/master/OSML+_on_server_with_Intel_Xeon_Gold_6338/)).