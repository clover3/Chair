import json
import logging
import os
import threading
import winsound

import requests

from cpath import data_path
from misc_lib import path_join, group_by


class NotificationHandler:
    def __init__(self, base_url, interval, send_os_notification):
        self.base_url = base_url
        self.interval = interval
        self.send_os_notification = send_os_notification
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_pool_action_periodically)
        self.sound_dir = path_join(data_path, "sound_data")
        self.tray_logger = logging.getLogger("Tray")
        self.tray_logger.setLevel(logging.DEBUG)

    def start(self):
        """Starts the periodic pooling."""
        self._thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        """Stops the periodic pooling."""
        self._stop_event.set()
        self._thread.join()

    def _run_pool_action_periodically(self):
        self.tray_logger.debug("_run_pool_action_periodically started")
        """Helper method to run pool_action periodically."""
        try:
            while not self._stop_event.is_set():
                self.tray_logger.info("Handler loop")
                self.pool_action()
                self._stop_event.wait(self.interval)
        except Exception as e:
            print(e)
            raise
        self.tray_logger.debug("_run_pool_action_periodically terminate")

    def clear_notification(self, notification):
        body = json.dumps({'id': notification['id']})
        response = requests.post(f'{self.base_url}/task/clear_notification', data=body)
        # Note: Handle the response if needed

    def play_sound(self, msg_type):
        if msg_type == "SUCCESSFUL_TERMINATE":
            file_name = "insanely_well_done.wav"
        elif msg_type == "ABNORMAL_TERMINATE":
            file_name = "14.wav"
        else:
            self.tray_logger.error("msg %s is not expected", msg_type)
            file_name = ""

        if file_name:
            sound_path = os.path.join(self.sound_dir, file_name)
            winsound.PlaySound(sound_path, winsound.SND_FILENAME)

    def handle_one_notification(self, notification):
        msg = notification["msg"]
        run_name = notification["task"]

        if "ABNORMAL_TERMINATE" in msg:
            self.play_sound("ABNORMAL_TERMINATE")
            self.send_os_notification(run_name, msg)

        if "SUCCESSFUL_TERMINATE" in msg:
            self.play_sound("SUCCESSFUL_TERMINATE")
            self.send_os_notification(run_name, msg)

        self.clear_notification(notification)

    def handle_notification_list(self, notification_list):
        g = group_by(notification_list, lambda x: x['msg'])
        for raw_msg, entries in g.items():
            msg_type = raw_msg.strip()
            task_name_list = [e["task"] for e in entries]
            display_msg = msg_type + "\n"
            if len(task_name_list) > 1:
                display_msg += "{} tasks completed. \n"

            display_msg += ", ".join(task_name_list)

            if "ABNORMAL_TERMINATE" in raw_msg:
                self.play_sound("ABNORMAL_TERMINATE")
                self.send_os_notification(msg_type, display_msg)

            if "SUCCESSFUL_TERMINATE" in raw_msg:
                self.play_sound("SUCCESSFUL_TERMINATE")
                self.send_os_notification(msg_type, display_msg)
        for notification in notification_list:
            self.tray_logger.info("handle notification %s", str(notification))
            self.clear_notification(notification)

    def pool_action(self):
        url = f'{self.base_url}/task/pool'
        response = requests.post(url, data=[])

        if response.status_code == 200:  # HTTP OK
            data = response.json()
            self.tray_logger.debug("data %s", str(data))
            notification_list = data.get('notifications', [])
            self.handle_notification_list(notification_list)
        else:
            self.tray_logger.info(response.content)