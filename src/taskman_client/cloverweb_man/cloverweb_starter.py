import subprocess
import time
import pandas
import io
from taskman_client.cloverweb_man.cloverweb_common import list_instances, KeepAlive, start_instance


def check_is_machine_off(inst_name):
    msg = list_instances()
    ret = pandas.read_fwf(io.StringIO(msg))
    try:
        status_dict = {}
        for row_idx, row_d in ret.to_dict('index').items():
            status_dict[row_d['NAME']] = row_d['STATUS']
        is_machine_off = status_dict[inst_name] == "TERMINATED"
    except KeyError as e:
        print(e)
        print("row_d", ret)
        print(ret.to_dict('index').items())
        is_machine_off= False
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


def loop():
    stop = False
    inst_name = "instance-1"
    keep_alive = KeepAlive(KEEP_ALIVE_INTERVAL)
    while not stop:
        # CHECK if GOSFORD is active
        if is_gosford_active():
            is_machine_off = check_is_machine_off(inst_name)
            if is_machine_off:
                tprint("Server is off. Starting {}".format(inst_name))
                stdout = start_instance(inst_name)
                tprint(stdout)
            else:
                keep_alive.send_keep_alive()
        else:
            tprint("Locked")
        time.sleep(CHECK_INTERVAL)


def main():
    loop()


if __name__ == "__main__":
    main()