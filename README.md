# OSML
OSML is an ML-based scheduler that intelligently schedules multiple interactive resources to meet co-located services' QoS targets. OSML employs multiple ML models to work collaboratively to predict QoS variations, shepherd the scheduling, and recover from QoS violations in complicated co-location cases.

## Pre-requirements and Benchmark Setup
1. Install Docker following instructions at [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/).

2. Install python dependencies.
```bash
python3 -m pip install -r requirements.txt
```

3. Pull the image containing benchmark applications from Docker Hub.
``` bash
docker pull douxl5516/osml_artifact:v0.92
```

4. Prepare input data for benchmark applications following instructions in [apps/README.md](https://github.com/Sys-Inventor-Lab/AI4System-OSML/blob/master/apps/README.md). We use volumes so that the applications in the docker container can read the input data from the host machine.

5. Install Intel's [pqos tool](https://github.com/intel/intel-cmt-cat).
```bash
cd thirdpart/
tar -xvf intel-cmt-cat.tar.gz
cd intel-cmt-cat/
make -j
make install
cd ../
```

6. Please note that some items in `configs.py` may need to be configured based on the server you are using, such as `MAX_LOAD`, `QOS_TARGET` and `RPS_COLLECTED`, which may vary on different servers.

## How to Run OSML
Run the following instruction.
```bash
python3 OSML.py
```

The OSML scheduler automatically launches the applications in the docker container according to `workload.txt` and schedules them. You can edit `workload.txt` to configure the applications to be launched. An example is given below. This example means that the scheduler will launch 3 applications including sphinx, img-dnn, specjbb, each with 30%, 30% and 90% of its maxload. Each application starts 48 (the number of logical cores on the platform, i.e., `N_CORES` in `configs.py`) threads. The RPS of sphinx is raised to 50% of its maxload at 20 seconds.
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
```

During the scheduling process, the OSML scheduler records the response latency and resource allocation of each application, and finally outputs them to the `logs/` directory.

## How to Obtain OSML Dataset
We have collected extensive real traces for widely deployed LC services on a platform equipped with an Intel Xeon E5-2697 v4 @ 2.3GHz CPU and 256GB memory. The traces are in a docker image. You can obtain the dataset using the following instructions. 
```
docker pull douxl5516/osml_dataset:v1.0
docker run -idt --name osml_dataset douxl5516/osml_dataset:v1.0 /bin/bash
docker cp osml_dataset:/home/data .
```

## How to Collect Your Own Dataset
1. Run `collect_single.py` (`collect_multiple.py`) in the `data/` directory to collect traces without (with) background applications.
   
2. Run `generate_dataset.sh` in the `data/` directory to label the raw data, and generated the dataset used for ML model training.

Data in the `data_collection` directory are raw data. Data in the `data_process` directory are processed dataset used for model training. Run `python count_samples.py` to see how many samples are covered by the dataset.

## How to Train and Test the ML Models
1. Run `Model_*.py` in the `models` directory to train and test the ML models using the generated dataset.
   
2. We provide well-trained ML models in the directory `models/`, facilitating transfer learning on new platforms.
