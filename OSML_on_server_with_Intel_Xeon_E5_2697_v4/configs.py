import os
import sys
import subprocess
import logging
import subprocess
from utils import shell_output
from enum import Enum

logger = logging.getLogger(__name__)
if not os.path.exists("logs"):
    outs, errs = shell_output("mkdir logs", wait = True, output = False)
logging.basicConfig(filename = "logs/osml.log", filemode = "w", level = logging.INFO, format='==> %(asctime)s - %(name)s[%(lineno)d] - %(levelname)s - %(message)s')
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# Mode of an application
# 0: Not launched
# 1: Unmanaged
# 2: Managed
# 3: Dead
# 4: Background
class MODE(Enum):
    Unlaunched = 0
    Unmanaged = 1
    Hungry = 2
    Managed = 3
    Dead = 4
    Background = 5


def init():
    global ROOT, PRIORITY, SHARING, MAX_INVOLVED, ACCEPTABLE_SLOWDOWN, SCHEDULING_INTERVAL, PQOS_OUTPUT_ENABLED, AGGRESSIVE, HISTORY_LEN, PERF_INTERVAL, TOP_INTERVAL, LATENCY_INTERVAL, NAMES, NAME_2_PNAME, QOS_TARGET, MAX_LOAD, RPS_COLLECTED, SETUP_STR, LAUNCH_STR, WARMUP_TIME, LATENCY_STR, CHANGE_RPS_STR, ACTION_SPACE, ACTION_ID, ACTION_SPACE_ADD, ACTION_SPACE_SUB, ACTION_ID_ADD, ACTION_ID_SUB, N_FEATURES, A_FEATURES, A_SHADOW_FEATURES, A_LABELS, B_FEATURES, B_LABELS, B_SHADOW_FEATURES, B_SHADOW_LABELS, C_FEATURES, COLLECT_FEATURES, COLLECT_MUL_FEATURES, COLLECT_N_FEATURES
    # Root path of the project
    ROOT = os.path.dirname(os.path.abspath(__file__))

    init_platform_conf()
    init_docker()

    # Prioritize the use of one type of resources
    # 0 for Default (Cores and LLC ways has the same priority)
    # 1 for Core (Cores are preferentially used)
    # 2 for Cache (LLC ways are preferentially used)
    PRIORITY = 0

    # Enable resource sharing
    # 0 for not enabled
    # 1 for enabled
    SHARING = 1

    # Maximum number of applications involved when enabling resource sharing or deprivation.
    MAX_INVOLVED = 3

    # Acceptable percentage of QoS slowdown. It is used when enabling Model-B or resource sharing.
    ACCEPTABLE_SLOWDOWN = 0.20

    # Scheduling interval (unit: s)
    SCHEDULING_INTERVAL = 1

    # Enable pqos output or not
    PQOS_OUTPUT_ENABLED = False

    # Enable aggressive deprivation
    AGGRESSIVE = True

    # Length of latency history and IPS history
    HISTORY_LEN = 5

    # Time interval between regular update sampling points of system performance monitor, in second
    PERF_INTERVAL = 0.1
    TOP_INTERVAL = 0.1
    LATENCY_INTERVAL = 0.5

    # Name of each application
    NAMES = ["img-dnn", "xapian", "moses", "sphinx", "specjbb",
             "masstree", "mongodb", "memcached", "login", "nginx", "ads", "mysql"]

    # Process name of each application
    NAME_2_PNAME = {"img-dnn": "img-dnn_integrated",
                    "xapian": "xapian_integrated",
                    "moses": "moses_integrated",
                    "sphinx": "decoder_integrated",
                    "specjbb": "java",
                    "masstree": "mttest_integrated",
                    "mongodb": "mongod",
                    "memcached": "memcached",
                    "login": "login_integrated",
                    "nginx": "nginx: worker process",
                    "ads": "ads_integrated",
                    "mysql": "mysqld",
                    "silo": "dbtest_integrated",
                    "redis": "redis-server",
                    "nodejs": "node"
                    }

    # QoS target of each application, in millisecond
    QOS_TARGET = {"img-dnn": 10,
                  "xapian": 10,
                  "moses": 10,
                  "sphinx": 3000,
                  "specjbb": 10,
                  "masstree": 10,
                  "mongodb": 1,
                  "memcached": 20,
                  "login": 5,
                  "nginx": 10,
                  "ads": 10,
                  "mysql": 5,
                  "silo": 10,
                  "ads": 40,
                  "redis": 0.5,
                  "nodejs": 200
                  }

    # Max load that can satisfy QoS target. Note that the max load may vary on different platforms.
    MAX_LOAD = {"img-dnn": 5500,
                "xapian": 8000,
                "moses": 1200,
                "sphinx": 15,
                "specjbb": 18000,
                "masstree": 1200,
                "mongodb": 9000,
                "memcached": 1280 * 1024,
                "login": 5500,
                "nginx": 300 * 1024,
                "ads": 700,
                "mysql": 190000,
                "silo": 2400,
                "redis": 66000,
                "nodejs": 1000}

    # RPSs used for data collection
    RPS_COLLECTED = {'img-dnn': [300, 800, 1300, 1800, 2300],
                     'xapian': [500, 1100, 1700, 2300, 2900],
                     "moses": [160, 320, 480, 640, 800],
                     'sphinx': [3, 6, 9, 12, 15],
                     'specjbb': [2800, 5600, 8400, 11200, 14000],
                     'masstree': [3000, 3400, 3800, 4200, 4600],
                     'login': [300, 600, 900, 1200, 1500],
                     'mongodb': [1000, 3000, 5000, 7000, 9000],
                     'memcached': [256 * 1024, 512 * 1024, 768 * 1024, 1024 * 1024, 1280 * 1024],
                     'nginx': [60 * 1024, 120 * 1024, 180 * 1024, 240 * 1024, 300 * 1024],
                     'ads': [10, 100, 1000]}

    SETUP_STR = {"mongodb": ["python2 /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/bin/ycsb load mongodb -s -P /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/workloads/test_load",
                             "mongod -f /etc/mongod.conf --repair"],
                 "mysql": ["python2 /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/bin/ycsb load jdbc -s -P /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/jdbc-binding/conf/db.properties -P /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/workloads/test_load_mysql"],
                 "redis": ["python2 /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/bin/ycsb load redis -s -P /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/workloads/test_load_redis"]
                 }

    # Instructions for launching each application
    LAUNCH_STR = {"img-dnn": "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/tailbench-v0.9/img-dnn/run.sh {RPS} 1440000 {threads}",
                  "xapian": "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/tailbench-v0.9/xapian/run.sh {RPS} 1632000 {threads}",
                  "moses": "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/tailbench-v0.9/moses/run.sh {RPS} 870000 {threads}",
                  "sphinx": "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/tailbench-v0.9/sphinx/run.sh {RPS} 9600 {threads}",
                  "specjbb": "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/tailbench-v0.9/specjbb/run.sh {RPS} 9000000 {threads}",
                  "masstree": "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/tailbench-v0.9/masstree/run.sh {RPS} 2760000 {threads}",
                  "mongodb": ["docker exec -itd " +  DOCKER_CONTAINER + " mongod -f /etc/mongod.conf --nojournal",
                              "docker exec -itd " +  DOCKER_CONTAINER + " mkdir -p /home/OSML_Artifact/apps/tmp/mongod_result/",
                              "docker exec -itd " +  DOCKER_CONTAINER + " bash -c 'python2 /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/bin/ycsb run mongodb -s -P /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/workloads/test -load -p status.interval=1 -p maxexecutiontime=600 -p target={RPS} -p threadcount={threads} 2>/home/OSML_Artifact/apps/tmp/mongod_result/ycsb_result'"],
                  "memcached": [
                      "docker exec -itd " +  DOCKER_CONTAINER + " memcached -d -m 40960 -u root -c 2048 -t {threads}",
                      "docker exec -itd " +  DOCKER_CONTAINER + " bash -c '/home/OSML_Artifact/apps/memcached/mutilate/mutilate -s 127.0.0.1:11211 -T "+str(N_CORES)+" -Q {RPS} -C 64 -t 600 -K 50 -V 400 > /home/OSML_Artifact/apps/tmp/memcached_lats_of_last_second.txt'"],
                  "login": "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/tailbench-v0.9/login/run.sh {RPS} 168000 {threads}",
                  "mysql": ["docker exec -itd " +  DOCKER_CONTAINER + " mysqld --basedir=/usr/local/mysql --datadir=/usr/local/mysql/data --plugin-dir=/usr/local/mysql/lib/plugin --user=root --log-error=obuntu213.err --pid-file=obuntu213.pid",
                            "docker exec -itd " +  DOCKER_CONTAINER + " mkdir -p /home/OSML_Artifact/apps/tmp/mysql_result/",
                            "docker exec -itd " +  DOCKER_CONTAINER + " bash -c 'python2 /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/bin/ycsb run jdbc -s -P /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/jdbc-binding/conf/db.properties -P /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/workloads/test_mysql -load -p status.interval=1 -p maxexecutiontime=600 -p target={RPS} -p threadcount={threads} 2>/home/OSML_Artifact/apps/tmp/mysql_result/ycsb_result'"],
                  "silo": "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/tailbench-v0.9/silo/run.sh {RPS} 1440000 {threads}",
                  "ads": "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/tailbench-v0.9/ads/run.sh {RPS} 600000 {threads}",
                  "nginx": ["docker exec -itd " +  DOCKER_CONTAINER + " cp /home/OSML_Artifact/apps/nginx/nginx.conf.default /usr/local/nginx/conf/nginx.conf",
                            "docker exec -itd " +  DOCKER_CONTAINER + " sed -i 's#worker_processes 1;#worker_processes {threads};#g' /etc/nginx/nginx.conf",
                            "docker exec -itd " +  DOCKER_CONTAINER + " nginx",
                            "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/nginx/wrk2/run.sh {RPS} 600"],
                  "redis": ["docker exec -itd " +  DOCKER_CONTAINER + " /usr/local/bin/redis-server /etc/redis/redis.conf",
                            "docker exec -itd " +  DOCKER_CONTAINER + " mkdir -p /home/OSML_Artifact/apps/tmp/redis_result/"
                            "docker exec -itd " +  DOCKER_CONTAINER + " bash -c 'python2 /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/bin/ycsb run redis -s -P /home/OSML_Artifact/apps/mongodb/ycsb-0.12.0/workloads/test_redis -load -p status.interval=1 -p maxexecutiontime=600 -p target={RPS} -p threadcount={threads} 2>/home/OSML_Artifact/apps/tmp/redis_result/ycsb_result'"],
                  "nodejs": ["docker exec -itd " +  DOCKER_CONTAINER + " node /home/OSML_Artifact/apps/nodejs/index.js",
                             "docker exec -itd " +  DOCKER_CONTAINER + " /home/OSML_Artifact/apps/nodejs/run_wrk.sh {RPS} 600"]
                  }

    # Warmup time after launching an application, in second
    WARMUP_TIME = {"img-dnn": 1,
                   "xapian": 1,
                   "moses": 10,
                   "sphinx": 1,
                   "specjbb": 1,
                   "masstree": 1,
                   "mongodb": 1,
                   "memcached": 1,
                   "login": 1,
                   "nginx": 1,
                   "silo": 20,
                   "mysql": 1,
                   "ads": 1,
                   "redis": 1,
                   "nodejs": 1}

    # Instructions for getting response latency of each application
    LATENCY_STR = {"img-dnn": "tail -n 1 "+VOLUME_PATH+"/tailbench-v0.9/img-dnn/latency_of_last_second.txt",
                   "xapian": "tail -n 1 "+VOLUME_PATH+"/tailbench-v0.9/xapian/latency_of_last_second.txt",
                   "moses": "tail -n 1 "+VOLUME_PATH+"/tailbench-v0.9/moses/latency_of_last_second.txt",
                   "specjbb": "tail -n 1 "+VOLUME_PATH+"/tailbench-v0.9/specjbb/latency_of_last_second.txt",
                   "masstree": "tail -n 1 "+VOLUME_PATH+"/tailbench-v0.9/masstree/latency_of_last_second.txt",
                   "sphinx": "tail -n 1 "+VOLUME_PATH+"/tailbench-v0.9/sphinx/latency_of_last_second.txt",
                   "login": "tail -n 1 "+VOLUME_PATH+"/tailbench-v0.9/login/latency_of_last_second.txt",
                   "nginx": "tail -n 1 "+VOLUME_PATH+"/nginx/RT/0.txt",
                   "mongodb": "tail -n 1 "+VOLUME_PATH+"/tmp/mongod_result/ycsb_result",
                   "mysql": "tail -n 1 "+VOLUME_PATH+"/tmp/mysql_result/ycsb_result",
                   "memcached": "tail -n 1 "+VOLUME_PATH+"/tmp/memcached_lats_of_last_second.txt",
                   "silo": "tail -n 1 "+VOLUME_PATH+"/tailbench-v0.9/silo/latency_of_last_second.txt",
                   "ads": "tail -n 1 "+VOLUME_PATH+"/tailbench-v0.9/ads/latency_of_last_second.txt",
                   "redis": "tail -n 1 "+VOLUME_PATH+"/tmp/redis_result/ycsb_result",
                   "nodejs": "tail -n 1 "+VOLUME_PATH+"/nodejs/RT/0.txt"}

    # Instructions for changing RPS of applications in tailbench
    CHANGE_RPS_STR = {"img-dnn": "echo {RPS} > "+VOLUME_PATH+"/tailbench-v0.9/img-dnn/RPS_NOW",
                      "xapian": "echo {RPS} > "+VOLUME_PATH+"/tailbench-v0.9/xapian/RPS_NOW",
                      "moses": "echo {RPS} > "+VOLUME_PATH+"/tailbench-v0.9/moses/RPS_NOW",
                      "specjbb": "echo {RPS} > "+VOLUME_PATH+"/tailbench-v0.9/specjbb/RPS_NOW",
                      "masstree": "echo {RPS} > "+VOLUME_PATH+"/tailbench-v0.9/masstree/RPS_NOW",
                      "sphinx": "echo {RPS} > "+VOLUME_PATH+"/tailbench-v0.9/sphinx/RPS_NOW",
                      "login": "echo {RPS} > "+VOLUME_PATH+"/tailbench-v0.9/login/RPS_NOW",
                      "silo": "echo {RPS} > "+VOLUME_PATH+"/tailbench-v0.9/silo/RPS_NOW",
                      "ads": "echo {RPS} > "+VOLUME_PATH+"/tailbench-v0.9/ads/RPS_NOW"}

    # Action space of Model-C, [delta core, delta way], e.g., [3,-1] represents allocating 3 cores and depriving 1 way.
    ACTION_SPACE =     [(-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3), 
                        (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3),
                        (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3),
                        (0, -3),  (0, -2),  (0, -1),  (0, 0),  (0, 1),  (0, 2),  (0, 3),
                        (1, -3), (1, -2), (1, -1), (1, 0), (1, 1),  (1, 2),  (1, 3),
                        (2, -3), (2, -2), (2, -1), (2, 0), (2, 1),  (2, 2),  (2, 3),
                        (3, -3), (3, -2), (3, -1), (3, 0), (3, 1),  (3, 2),  (3, 3)]
    ACTION_ID = {action:_ for _,action in enumerate(ACTION_SPACE)}

    ACTION_SPACE_ADD = [(-3, 1), (-3, 2), (-3, 3),
                        (-2, 1), (-2, 2), (-2, 3),
                        (-1, 1), (-1, 2), (-1, 3),
                        (0, 0), (0, 1),  (0, 2),  (0, 3),
                        (1, -3), (1, -2), (1, -1), (1, 0), (1, 1),  (1, 2),  (1, 3),
                        (2, -3), (2, -2), (2, -1), (2, 0), (2, 1),  (2, 2),  (2, 3),
                        (3, -3), (3, -2), (3, -1), (3, 0), (3, 1),  (3, 2),  (3, 3)]

    ACTION_SPACE_SUB = [(-3, -3), (-3, -2), (-3, -1), (-3, 0),
                        (-2, -3), (-2, -2), (-2, -1), (-2, 0),
                        (-1, -3), (-1, -2), (-1, -1), (-1, 0),
                        (0, -3),  (0, -2),  (0, -1),  (0, 0)]

    ACTION_ID_ADD = {action: _ for _, action in enumerate(ACTION_SPACE_ADD)}
    ACTION_ID_SUB = {action: _ for _, action in enumerate(ACTION_SPACE_SUB)}

    N_FEATURES = ["MBL_N", "Allocated_Cache_N", "Allocated_Core_N"]

    # Features of Model-A
    A_FEATURES = ["CPU_Utilization", "Frequency", "IPC", "Misses", "MBL",
                  "Virt_Memory", "Res_Memory", "Allocated_Cache", "Allocated_Core"]

    # Features of Model-A' (shadow of Model-A)
    A_SHADOW_FEATURES = ["CPU_Utilization", "Frequency", "IPC", "Misses", "MBL", "Virt_Memory",
                         "Res_Memory", "Allocated_Cache", "Allocated_Core", "MBL_N", "Allocated_Cache_N", "Allocated_Core_N"]

    # Labels of Model-A and Model-A'
    if MBA_SUPPORT:
        A_LABELS = ["RCliff_Cache", "RCliff_Core",
                    "OAA_Cache", "OAA_Core", "OAA_Bandwidth"]
    else:
        A_LABELS = ["RCliff_Cache", "RCliff_Core", "OAA_Cache", "OAA_Core"]

    # Features of Model-B
    B_FEATURES = ["CPU_Utilization", "Frequency", "IPC", "Misses", "MBL", "Virt_Memory", "Res_Memory",
                  "Allocated_Cache", "Allocated_Core", "MBL_N", "Allocated_Cache_N", "Allocated_Core_N", "QoS_Reduction"]

    # Labels of Model-B
    B_LABELS = ["Core_Dominated_Spared_Cache", "Core_Dominated_Spared_Core", "Default_Spared_Cache",
                "Default_Spared_Core", "Cache_Dominated_Spared_Cache", "Cache_Dominated_Spared_Core"]

    # Features of Model-B' (shadow of Model-B)
    B_SHADOW_FEATURES = ["CPU_Utilization", "Frequency", "IPC", "Misses", "MBL", "Virt_Memory", "Res_Memory",
                         "Allocated_Cache", "Allocated_Core", "MBL_N", "Allocated_Cache_N", "Allocated_Core_N", "Target_Cache", "Target_Core"]

    # Labels of Model-B'
    B_SHADOW_LABELS = ["QoS_Reduction"]

    # Features of Model-C
    C_FEATURES = {"s": ["CPU_Utilization", "Frequency", "IPC", "Misses", "MBL", "Allocated_Cache", "Allocated_Core", "Latency"],
                  "a": ["Action_ID"],
                  "r": ["Reward"],
                  "s_": ["CPU_Utilization_", "Frequency_", "IPC_", "Misses_", "MBL_", "Allocated_Cache_", "Allocated_Core_", "Latency_"],
                  "best_a": ["Best_Action_ID"]}

    # Features for data collection when only one program is running
    COLLECT_FEATURES = ['CPU_Utilization', 'Frequency', 'IPC', 'Misses', 'MBL',
                        'Virt_Memory', 'Res_Memory', 'Allocated_Cache', 'Allocated_Core', 'Latency']

    # Features for data collection when multiple programs are running
    COLLECT_MUL_FEATURES = ["CPU_Utilization", "Frequency", "IPC", "Misses", "MBL", "Virt_Memory", "Res_Memory", "Allocated_Cache", "Allocated_Core", "MBL_N", "Allocated_Cache_N", "Allocated_Core_N", "Latency"]

    # Features collected from Neighbors
    COLLECT_N_FEATURES = ["MBL", "Allocated_Cache", "Allocated_Core"]


def init_docker():
    global DOCKER_IMAGE, DOCKER_CONTAINER, BIND_PATH, VOLUME_PATH
    DOCKER_IMAGE = "sysinventor/osml_benchmark:v1.0"
    DOCKER_CONTAINER = "benchmark_container"
    BIND_PATH = None
    VOLUME_PATH = None
    # Start benchmark_container and get the volume path
    outs, errs = shell_output("docker pull {}".format(DOCKER_IMAGE),wait=True,output=False)
    outs, errs = shell_output("docker run -idt -v {}/volume:/home/OSML_Artifact/volume:rw -v /home/OSML_Artifact/apps --name {} {} /bin/bash".format(ROOT,DOCKER_CONTAINER, DOCKER_IMAGE), wait=True, output=False)
    outs, errs = shell_output("docker start {}".format(DOCKER_CONTAINER), wait=True, output=False)
    inspect_info = eval(subprocess.check_output("docker inspect {}".format(DOCKER_CONTAINER), shell=True).decode().replace("false", "False").replace("null", "None").replace("true", "True"))
    mount_info = inspect_info[0]["Mounts"]

    for item in mount_info:
        if item["Type"] == "volume":
            VOLUME_PATH = item["Source"]
        elif item["Type"] == "bind":
            BIND_PATH = item["Source"]

    def init_tailbench():
        if not os.path.exists(BIND_PATH+"/tailbench.inputs"):
            raise Exception("Please download the input of tailbench and put the \"tailbench.inputs\" folder in {}".format(BIND_PATH))

    def init_nginx():
        # Set this to point to the top level of the Nginx data directory
        PATH_NGINX_INPUTS = ROOT+"/apps/nginx/html/"
        # Prepare inputs for Nginx
        if not os.path.exists(PATH_NGINX_INPUTS):
            print("Generating inputs for Nginx, wait a moment.")
            os.system("sudo {}/apps/nginx/gen_html.sh".format(ROOT))
        if not os.path.exists(BIND_PATH+"/html"):
            print("Copying html to docker volume, wait a moment.")
            os.system("sudo cp -r /dev/shm/html "+BIND_PATH+"/html")
            os.system("docker exec -itd "+DOCKER_CONTAINER+" cp -r /home/OSML_Artifact/volume/html /dev/shm/")

    init_tailbench()
    init_nginx()


def init_platform_conf():
    global N_CORES, N_WAYS, MB_PER_WAY, N_COS, MBA_SUPPORT, MAX_THREADS, CORE_INDEX, WAY_INDEX, PHYSICAL_CORES
    core_info_str = [line.strip() for line in subprocess.check_output("pqos -I -s | grep 'Core'", shell=True).decode().split("\n")]
    core_info = []
    for line in core_info_str:
        if line == "" or line.startswith("Core information"):
            continue
        arr = line.split()
        core_idx = int(arr[1].rstrip(","))
        L2_idx = int(arr[3].rstrip(","))
        L3_idx = int(arr[5])
        core_info.append((core_idx, L2_idx, L3_idx))
    core_info.sort(key=lambda x: (x[2], x[1], x[0]))
    N_CORES = len([item for item in core_info if item[2] == 0])
    CORE_INDEX = list(range(0, N_CORES))
    # List of Physical cores. The hyper-threading is enabled. Two logical cores share the same physical core. e.g., logical cores with index 0 and 18 are on one physical core, they share the L1 and L2 cache.
    PHYSICAL_CORES = [item[0] for item in core_info if item[2] == 0]
    N_WAYS = int(subprocess.check_output(
        "cat /sys/devices/system/cpu/cpu0/cache/index3/ways_of_associativity", shell=True))
    MB_PER_WAY = int(subprocess.check_output(
        "cat /sys/devices/system/cpu/cpu0/cache/index3/size", shell=True).decode().rstrip("K\n")) / 1024 / N_WAYS
    N_COS = int(subprocess.check_output(
        "pqos -I -d | grep COS | awk {'print $3'}", shell=True).split()[0])
    MBA_SUPPORT = False
    MAX_THREADS = N_CORES
    WAY_INDEX = list(range(N_WAYS))

init()
