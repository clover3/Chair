import os
import logging
import subprocess
import sys
import time

import requests
import urllib3

logger = None


def init_log():
    global logger
    logger = logging.getLogger('Cloverweb')
    logger.setLevel(logging.INFO)
    format_str = '%(levelname)s\t%(name)s \t%(asctime)s %(message)s'
    formatter = logging.Formatter(format_str,
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(ch)


def tprint(msg):
    global logger
    if logger is None:
        init_log()

    logger.info(msg)


def execute_gcloud_command(cmd):
    proc = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    return proc.stdout + proc.stderr


def start_instance(inst_name) -> str:
    execute_gcloud_command("gcloud config set project cloverweb")
    cmd = "gcloud compute instances start {} --zone=us-central1-a".format(inst_name)
    return execute_gcloud_command(cmd)


def list_instances() -> str:
    execute_gcloud_command("gcloud config set project cloverweb")
    cmd = "gcloud compute instances list"
    return execute_gcloud_command(cmd)


def time_since(last_time):
    now = time.time()
    if last_time is None:
        return now
    else:
        return now - last_time


class KeepAlive:
    def __init__(self, interval):
        self.last_request = None
        self.interval = interval

    def send_keep_alive(self):
        elapsed = time_since(self.last_request)
        if elapsed >= self.interval:
            tprint("Send keep alive")
            server_url = "http://clovertask2.xyz/"
            try:
                receive = requests.get(server_url)
                if receive.status_code != 200:
                    print(receive.status_code)
                    print(receive.content)
            except urllib3.exceptions.NewConnectionError as e:
                print(e)
            self.last_request = time.time()
        else:
            tprint("Skip keep alive")


def parse_ip(s: str):
    pattern = "Instance external IP is "
    ip = None
    for line in s.split("\n"):
        if line.startswith(pattern):
            ip = line[len(pattern):].strip()
            break
    if ip is None:
        raise Exception()

    return ip


def replace_sync_script(file_path, ip):
    def replace_line(line):
        if "@" in line and ":" in line:
            name_at_server, dir_path = line.split(":")
            name, server = name_at_server.split("@")
            tprint(name, server, dir_path)
            return "{}@{}:{}".format(name, ip, dir_path) + "\n"
        else:
            return line

    lines = open(file_path).readlines()
    new_lines = map(replace_line, lines)
    open(file_path, "w").writelines(new_lines)


def replace_file(file_path, ip):
    lines = open(file_path, encoding="utf16").readlines()

    def replace_line(line):
        pattern = "Host="
        if line.startswith(pattern):
            line = pattern + ip + "\n"
        return line

    new_lines = map(replace_line, lines)
    open(file_path, "w", encoding="utf16").writelines(new_lines)
