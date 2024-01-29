import time
import numpy as np
from copy import deepcopy

from utils import *
from configs import *
from program_mgr import *


def OSML(mgr, terminate_when_QoS_is_met=False, terminate_when_timeout=False, timeout=300):
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

            if not mgr.log_thread_configs["running"]:
                mgr.start_log_thread()

            if mgr.size_onfly() == 1:
                # When only one program is running, use Model-A
                A_points = mgr.use_model_A(app)
            else:
                # When multiple programs are running, use Model-A'
                A_points = mgr.use_model_A_shadow(app)

            idle = mgr.resource_idle(exclude=app)
            A_solutions = filter_A_solutions(A_points, idle, mgr)
            if len(A_solutions) > 0:
                # Idle resources are sufficient to meet OAA/RCliff.
                if "OAA" in A_solutions:
                    mgr.allocate(app, A_solutions["OAA"])
                    mgr.report_allocation()
                    # print_color("Idle resources are sufficient to meet OAA.", "green")
                elif "RCliff" in A_solutions:
                    mgr.allocate(app, {"cores": min(idle["cores"], A_points["OAA"]["cores"]), "ways": min(
                        idle["ways"], A_points["OAA"]["ways"])})
                    mgr.report_allocation()
                    # print_color("Idle resources are sufficient to meet RCliff.", "green")
            else:
                # Idle resources are not enough, Enable Model-B
                diffs = {case: {"cores": A_points[case]["cores"] - idle["cores"],
                                "ways": A_points[case]["ways"] - idle["ways"]} for case in ["RCliff", "OAA"]}
                B_points = mgr.use_model_B(mgr.neighbors(app), ACCEPTABLE_SLOWDOWN)
                # print("B_points:", B_points)
                B_solutions = filter_B_solutions(B_points, diffs)
                # print("B_solutions:", B_solutions)
                if len(B_solutions) > 0:
                    for case in B_solutions:
                        B_solution = B_solutions[case][0]
                        diff = deepcopy(diffs[case])
                        deprivation_policy = generate_deprivation_policy(
                            B_solution, diff, mgr, app)
                        # print("deprivation_policy", deprivation_policy)
                        for key in deprivation_policy:
                            mgr.allocate_diff(key[0], {"cores": -deprivation_policy[key]["cores"],
                                                       "ways": -deprivation_policy[key]["ways"]}, propagate=False)

                        idle = mgr.resource_idle(exclude=app)
                        mgr.allocate(app, {"cores": min(idle["cores"], A_points[case]["cores"]), "ways": min(
                            idle["ways"], A_points[case]["ways"])}, propagate=True)
                        mgr.report_allocation()
                        break
                elif SHARING:
                    # Model-B failed, try resource sharing among applications
                    # print_color("Model-B failed. Try resource sharing among applications.", "cyan")
                    max_diff = {"cores": max([diffs[key]["cores"] for key in diffs]),
                                "ways": max([diffs[key]["ways"] for key in diffs])}
                    sharing_policies = []
                    for n_cores, n_ways in product(list(range(max_diff["cores"] + 1)),
                                                   list(range(max_diff["ways"] + 1))):
                        sharing_policies.append(
                            {"cores": int(n_cores), "ways": int(n_ways)})
                    B_shadow_points = mgr.use_model_B_shadow(
                        mgr.neighbors(app), sharing_policies)
                    B_shadow_solutions = filter_B_shadow_solutions(
                        B_shadow_points, diffs, mgr)
                    if len(B_shadow_solutions) > 0:
                        for case in B_shadow_solutions:
                            B_shadow_solution = B_shadow_solutions[case][0]
                            diff = diffs[case]
                            deprivation_policy = generate_deprivation_policy(
                                B_shadow_solution, diff, mgr, app)

                            for key in deprivation_policy:
                                mgr.allocate_diff(key[0], {"cores": -deprivation_policy[key]["cores"],
                                                           "ways": -deprivation_policy[key]["ways"]}, propagate=False)

                            apps = [key[0]
                                    for key in deprivation_policy.keys()]
                            apps.append(app)
                            mgr.allocate_sharing(apps, {"cores": sum([deprivation_policy[key]["cores"] for key in deprivation_policy]), "ways": sum(
                                [deprivation_policy[key]["ways"] for key in deprivation_policy])}, propagate=True)
                            mgr.report_allocation()
                            break
                    else:
                        # Allocate all idle resources to this application
                        core_len = min(
                            idle["cores"], A_points["RCliff"]["cores"])
                        way_len = min(idle["ways"], A_points["RCliff"]["ways"])
                        cores_required = 0 if core_len > 0 else 1
                        ways_required = 0 if way_len > 0 else 1

                        if not (cores_required == 0 and ways_required == 0):
                            victim = self.select_victim(
                                cores_required, ways_required)
                            mgr.allocate_diff(
                                victim, {"cores": -1*cores_required, "ways": -1*ways_required})

                        mgr.allocate(
                            app, {"cores": core_len+cores_required, "ways": way_len+ways_required})

        mgr.report_latency()
        mgr.report_allocation()

        mgr.check_revert_event()
        mgr.process_last_model_C_action()

        under_provision_apps = mgr.get_under_provision_apps()
        over_provision_apps = mgr.get_over_provision_apps()
        under_provision_QoS_violations = {app: mgr.get_QoS_violation(app) for app in under_provision_apps}
        over_provision_QoS_violations = {app: mgr.get_QoS_violation(app) for app in over_provision_apps}
        under_provision_apps.sort(key=lambda x: under_provision_QoS_violations[x], reverse=True)
        over_provision_apps.sort(key=lambda x: over_provision_QoS_violations[x])

        if any([mgr.get_QoS_violation(name) >= 2 for name in mgr.programs]):
            if len(under_provision_apps) > 0:
                # print_color("Under provision apps: {}".format(", ".join(under_provision_apps)), "cyan")
                mgr.use_model_C_add(under_provision_apps[0])

            if len(over_provision_apps) > 0 and len(under_provision_apps) > 0:
                # print_color("Over provision apps: {}".format(", ".join(over_provision_apps)), "cyan")
                mgr.use_model_C_sub(over_provision_apps[0])

        if terminate_when_QoS_is_met and mgr.is_all_QoS_met() and time.time()-mgr.QoS_met_time > 10:
            print_color("Terminate because the QoS is met.", "green")
            time.sleep(5)  # record latency
            return

        if terminate_when_timeout and time.time()-mgr.start_time > timeout:
            print_color("Terminate because the time is out.", "green")
            return

        time_remaining = SCHEDULING_INTERVAL - time.time() % SCHEDULING_INTERVAL
        time.sleep(time_remaining)


def main():
    mgr = program_mgr(config_path=ROOT + "/workload.txt", regular_update=True)
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
