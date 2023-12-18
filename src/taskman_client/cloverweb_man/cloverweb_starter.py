import io
import logging
import subprocess
import time

import pandas

from taskman_client.cloverweb_man.cloverweb_common import list_instances, KeepAlive, start_instance


def check_is_machine_off(inst_name, clover_web_logger):
    msg = list_instances()
    ret = pandas.read_fwf(io.StringIO(msg))
    try:
        status_dict = {}
        for row_idx, row_d in ret.to_dict('index').items():
            status_dict[row_d['NAME']] = row_d['STATUS']
        is_machine_off = status_dict[inst_name] == "TERMINATED"
    except KeyError as e:
        clover_web_logger.warning(e)
        clover_web_logger.warning("row_d %s", str(ret))
        clover_web_logger.warning(str(ret.to_dict('index').items()))
        is_machine_off = False
    return is_machine_off


def is_gosford_active():
    proc = subprocess.run("tasklist", shell=True, capture_output=True, encoding="utf-8")
    ret = pandas.read_fwf(io.StringIO(proc.stdout))
    names = set(ret['Image Name'])
    if 'LogonUI.exe' in names:
        active = False
    elif 'logon.scr' in names:
        active = False
    else:
        active = True
    return active


CHECK_INTERVAL = 20
KEEP_ALIVE_INTERVAL = 120


def keep_server_alive_loop(f_stop_fn=None):
    clover_web_logger = logging.getLogger('Tray')
    clover_web_logger.setLevel(logging.INFO)
    stop = False
    inst_name = "instance-1"
    keep_alive = KeepAlive(KEEP_ALIVE_INTERVAL, clover_web_logger)
    while not stop:
        print("keep_server_alive_loop")
        # CHECK if GOSFORD is active
        if is_gosford_active():
            is_machine_off = check_is_machine_off(inst_name, clover_web_logger)
            if is_machine_off:
                clover_web_logger.info("Server is off. Starting {}".format(inst_name))
                stdout = start_instance(inst_name)
                clover_web_logger.info(stdout)
            else:
                keep_alive.send_keep_alive()
        else:
            clover_web_logger.debug("Locked")
        time.sleep(CHECK_INTERVAL)

        if f_stop_fn is not None:
            stop = f_stop_fn()


def main():
    keep_server_alive_loop()


if __name__ == "__main__":
    main()
