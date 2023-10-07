import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSystemTrayIcon, QMenu,
                             QAction, QVBoxLayout, QPushButton, QLabel, QWidget, QStyle)
from PyQt5.QtGui import QIcon
import json
import requests
import threading


class NotificationHandler:

    def __init__(self, base_url, interval):
        self.base_url = base_url
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_pool_action_periodically)

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
            self.pool_action()
            self._stop_event.wait(self.interval)

    def clear_notification(self, notification):
        body = json.dumps({'id': notification['id']})
        response = requests.post(f'{self.base_url}/task/clear_notification', data=body)
        # Note: Handle the response if needed

    def handle_notification(self, notification):
        self.clear_notification(notification)

    def pool_action(self):
        response = requests.post(f'{self.base_url}/task/pool', data=[])

        if response.status_code == 200:  # HTTP OK
            data = response.json()
            for notification in data.get('notifications', []):
                self.handle_notification(notification)
        else:
            print(response.content)



class TrayApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # handler = NotificationHandler("http://your_domain_or_ip", 10)  # Pooling every 10 seconds

        # Set window title and initial size
        self.setWindowTitle("Tray Application")
        self.setGeometry(100, 100, 400, 300)

        # Create a system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))

        # UI elements for the main window
        self.layout = QVBoxLayout()

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

    def closeEvent(self, event):
        event.ignore()
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
