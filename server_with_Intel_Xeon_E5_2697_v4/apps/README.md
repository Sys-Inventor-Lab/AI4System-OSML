# Workload Setup

We have uploaded the compiled workload applications into a Docker image. You can pull the image by running the following command.
``` bash
docker pull sysinventor/osml_artifact:v0.92
```

To run Tailbench, MongoDB and Nginx, you need to follow the instructions below to prepare input data for these applications.

## Tailbench

We use 6 applications in [Tailbench](http://tailbench.csail.mit.edu/), including Img-dnn, Masstree, Moses, Specjbb, Sphinx, and Xapian. We also implement Login and Ads based on the framework of Tailbench. We use the integrated version of Tailbench.

#### New features added
We've made changes to Tailbench, adding two new features:

1. Each application can output the per-second latency to a file named `latency_of_last_second.txt` in `$PWD`.

2. The request per second(RPS) of an application can be dynamically changed by writing the RPS value to a file named `RPS_NOW`.

#### How to run Tailbench
Download TailBench data [here](http://tailbench.csail.mit.edu/tailbench.inputs.tgz). Place the decompressed data in the `volume/` directory of the host machine.

## MongoDB
We use ycsb as the load generator. Run following scripts in the bash of the docker container with a name of `workload_container` to preload the test workload.
```bash
mongod -f /etc/mongod.conf &
python2 /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/bin/ycsb load mongodb -s -P /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/workloads/test_load
```

## Nginx
Generate random html files as Nginx's dataset in the host machine.
```bash
cd apps/nginx/
./gen_html.sh
```
