import time
import numpy as np
from copy import deepcopy

from utils import *
from configs import *
from program_mgr import *


def OSML(mgr, terminate_when_QoS_is_met=True, terminate_when_timeout=True, timeout=180):
    while True:
        if mgr.all_done():
            break
        for app in list(mgr.programs.keys()):
            if mgr.can_be_ended(app):
                mgr.end(app)
            if mgr.RPS_can_be_changed(app):
                mgr.change_RPS(app)

        mgr.update_pending_queue()

        # Launch all applications in the pending queue
        for app in list(mgr.pending_queue.keys()):
            if not mgr.can_be_launched(app):
                continue
            mgr.launch(app)

            # Start log threads
            if not mgr.log_thread_configs["running"]:
                mgr.start_log_thread()
            
            # Allocate resources for the newly started application
            A_points = mgr.use_model_A(app)
            idle = mgr.resource_idle(exclude=app)
            if A_points["OAA"]["cores"]<=idle["cores"] and A_points["OAA"]["ways"]<=idle["ways"]:
                A_solution=A_points["OAA"]
            else:
                A_solution=A_points["RCliff"]
            diff={"cores":A_solution["cores"]-mgr.programs[app].core_len,"ways":A_solution["ways"]-mgr.programs[app].way_len}
            res=mgr.adjust_using_model_B(app,diff)

        mgr.report_latency()
        mgr.report_allocation()

        mgr.check_revert_event()

        under_provision_apps = mgr.get_under_provision_apps()
        over_provision_apps = mgr.get_over_provision_apps()

        under_provision_slack = {app: mgr.get_slack(app) for app in under_provision_apps}
        over_provision_slack = {app: mgr.get_slack(app) for app in over_provision_apps}

        under_provision_apps.sort(key=lambda x: under_provision_slack[x], reverse=True)
        over_provision_apps.sort(key=lambda x: over_provision_slack[x])

        for app in under_provision_apps:
            mgr.use_model_C(app)
        for app in over_provision_apps:
            mgr.use_model_C(app)

        if terminate_when_QoS_is_met and mgr.is_all_QoS_met() and time.time()-mgr.QoS_met_time > 10:
            print_color("Terminate because the QoS is met.", "green")
            logger.info("Terminate because the QoS is met.")
            time.sleep(5)  # record latency
            return

        if terminate_when_timeout and time.time()-mgr.start_time > timeout:
            print_color("Terminate because the time is out.", "green")
            logger.info("Terminate because the time is out.")
            return

        time_remaining = SCHEDULING_INTERVAL - time.time() % SCHEDULING_INTERVAL
        time.sleep(time_remaining)



def main():
    mgr = program_mgr(config_path=ROOT + "/workload.txt", regular_update=True, enable_models=True, training=True)
    try:
        OSML(mgr)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        mgr.end_log_thread()
        mgr.end_all()
        raise e
    mgr.end_log_thread()
    mgr.end_all()
    os.system("./reset.sh")



if __name__ == '__main__':
    main()
