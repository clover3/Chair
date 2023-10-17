import json
import logging
import os
import sys
import threading
import winsound

import requests
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSystemTrayIcon, QMenu,
                             QAction, QVBoxLayout, QPushButton, QLabel, QWidget, QStyle)

from cpath import data_path
from misc_lib import path_join
from trainer_v2.chair_logging import c_log


class NotificationHandler:
    def __init__(self, base_url, interval):
        self.base_url = base_url
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_pool_action_periodically)
        self.sound_dir = path_join(data_path, "sound_data")

    def start(self):
        """Starts the periodic pooling."""
        self._thread.start()

    def stop(self):
        """Stops the periodic pooling."""
        self._stop_event.set()
        self._thread.join()

    def _run_pool_action_periodically(self):
        """Helper method to run pool_action periodically."""
        while not self._stop_event.is_set():
            c_log.info("Handler loop")
            self.pool_action()
            self._stop_event.wait(self.interval)

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
            c_log.error("msg %s is not expected", msg_type)
            file_name = ""

        if file_name:
            sound_path = os.path.join(self.sound_dir, file_name)
            winsound.PlaySound(sound_path, winsound.SND_FILENAME)

    def handle_notification(self, notification):
        msg = notification["msg"]
        if "ABNORMAL_TERMINATE" in msg:
            self.play_sound("ABNORMAL_TERMINATE")
        if "SUCCESSFUL_TERMINATE" in msg:
            self.play_sound("SUCCESSFUL_TERMINATE")

        self.clear_notification(notification)

    def pool_action(self):
        url = f'{self.base_url}/task/pool'
        response = requests.post(url, data=[])

        if response.status_code == 200:  # HTTP OK
            data = response.json()
            c_log.debug("data %s", str(data))
            for notification in data.get('notifications', []):
                c_log.info("handle notification %s", str(notification))
                self.handle_notification(notification)
        else:
            print(response.content)


class TrayApp(QMainWindow):
    def __init__(self, domain="https://clovertask2.xyz:8000"):
        super().__init__()
        self.handler = NotificationHandler(domain, 10)  # Pooling every 10 seconds

        # Set window title and initial size
        self.setWindowTitle("Tray Application")
        self.setGeometry(100, 100, 400, 300)

        # Create a system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        # icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        icon = QIcon(path_join(data_path, "html", "task.png"))
        self.tray_icon.setIcon(icon)


        # UI elements for the main window
        self.layout = QVBoxLayout()

        self.tray_button = QPushButton("Put to tray", self)
        self.tray_button.clicked.connect(self.put_to_tray)
        self.layout.addWidget(self.tray_button)

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_app)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_app)
        self.layout.addWidget(self.stop_button)

        self.status_label = QLabel("Status: Stopped", self)
        self.layout.addWidget(self.status_label)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        # Create a context menu for the tray
        tray_menu = QMenu()

        # Create a "Show" action
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)

        # Create a "Quit" action
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close_app)
        tray_menu.addAction(quit_action)

        # Set the context menu to the tray icon
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def put_to_tray(self):
        self.hide()
        self.tray_icon.showMessage(
            "Tray App",
            "App minimized to tray. Right-click on the icon to show or quit.",
            QSystemTrayIcon.Information
        )

    def start_app(self):
        # Implement your start logic here
        self.status_label.setText("Pooler status: Started")
        self.handler.start()

    def stop_app(self):
        # Implement your stop logic here
        self.status_label.setText("Pooler Status: Stopped")
        self.handler.stop()

    def close_app(self):
        self.tray_icon.hide()
        QApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrayApp()
    window.show()
    sys.exit(app.exec_())
