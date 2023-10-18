import json
import logging
import os
import sys
import threading
import winsound
from plyer import notification

import requests
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSystemTrayIcon, QMenu,
                             QAction, QVBoxLayout, QPushButton, QLabel, QWidget, QStyle)

from cpath import data_path
from misc_lib import path_join
from trainer_v2.chair_logging import c_log


class NotificationHandler:
    def __init__(self, base_url, interval, send_os_notification):
        self.base_url = base_url
        self.interval = interval
        self.send_os_notification = send_os_notification
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
            c_log.debug("Handler loop")
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
        run_name = notification["task"]

        if "ABNORMAL_TERMINATE" in msg:
            self.play_sound("ABNORMAL_TERMINATE")
            self.send_os_notification(run_name, msg)

        if "SUCCESSFUL_TERMINATE" in msg:
            self.play_sound("SUCCESSFUL_TERMINATE")
            self.send_os_notification(run_name, msg)

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
            c_log.info(response.content)


class TrayApp(QMainWindow):
    def __init__(self, domain="https://clovertask2.xyz:8000"):
        super().__init__()

        # Set window title and initial size
        self.app_name = "Taskman Daemon"
        self.handler = NotificationHandler(domain, 10, self.send_os_notification)  # Pooling every 10 seconds
        self.setWindowTitle(self.app_name)
        self.setGeometry(100, 100, 400, 300)

        # Create a system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        # icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        self.icon_path = path_join(data_path, "html", "task.png")
        icon = QIcon(self.icon_path)
        self.tray_icon.setIcon(icon)
        self.setWindowIcon(icon)

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
        self.stop_button.setEnabled(False)

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

        self.tray_icon.activated.connect(self.on_tray_icon_activated)

        # Set the context menu to the tray icon
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def send_os_notification(self, title, message):
        notification.notify(
            title=title,
            message=message,
            app_name=self.app_name,
            app_icon=self.icon_path,
            timeout=10  # the notification will stay for 10 seconds
        )

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
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.handler.start()

    def stop_app(self):
        # Implement your stop logic here
        self.status_label.setText("Pooler Status: Stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.handler.stop()

    def close_app(self):
        self.tray_icon.hide()
        QApplication.quit()

    # Our custom slot that handles the tray icon activation
    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:  # Tray icon was clicked
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()  # Bring window to front


if __name__ == '__main__':
    c_log.setLevel(logging.INFO)
    app = QApplication(sys.argv)
    window = TrayApp()
    window.show()
    sys.exit(app.exec_())
