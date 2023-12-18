import logging
import sys
from PyQt5.QtCore import QObject, pyqtSignal
from plyer import notification

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSystemTrayIcon, QMenu,
                             QAction, QVBoxLayout, QPushButton, QLabel, QWidget, QTextEdit)
import ctypes

from cpath import data_path
from misc_lib import path_join
from taskman_client.cloverweb_man.cloverweb_starter import keep_server_alive_loop
from taskman_client.cloverweb_man.notification_handler import NotificationHandler
import threading

from taskman_client.cloverweb_man.tray_logger_module import tray_logger


class QTextEditLoggingHandler(logging.Handler, QObject):
    append_signal = pyqtSignal(str)

    def __init__(self, widget):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.widget = widget
        self.widget.setReadOnly(True)
        self.append_signal.connect(self.widget.append)

    def emit(self, record):
        msg = self.format(record)
        self.append_signal.emit(msg)


def terminate_thread(thread):
    """Terminates a python thread from another thread.

    :param thread: a threading.Thread instance
    """

    print("terminate_thread")
    if not thread.isAlive():
        return

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


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

        self.log_display = QTextEdit(self)
        self.layout.addWidget(self.log_display)
        logging.info("Logging before set up")
        self.setup_logging()
        tray_logger.info("Logging after set up")

        self.tray_icon.activated.connect(self.on_tray_icon_activated)

        # Set the context menu to the tray icon
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        self._stop_event = threading.Event()
        self._server_starter_thread = threading.Thread(target=keep_server_alive_loop, args=(self.f_stop,))
        self._server_starter_thread.start()

    def f_stop(self):
        return self._stop_event.is_set()

    def setup_logging(self):
        tray_logger = logging.getLogger("Tray")
        tray_logger.setLevel(logging.INFO)
        format_str = '%(levelname)s %(name)s %(asctime)s %(message)s'
        formatter = logging.Formatter(format_str,
                                      datefmt='%m-%d %H:%M:%S',
                                      )

        logTextBox = QTextEditLoggingHandler(self.log_display)
        logTextBox.setLevel(logging.INFO)
        logTextBox.setFormatter(formatter)
        tray_logger.addHandler(logTextBox)


        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.DEBUG)

        clover_web_logger = logging.getLogger('CloverWeb')
        clover_web_logger.setLevel(logging.DEBUG)
        logTextBox.setLevel(logging.INFO)
        clover_web_logger.addHandler(logTextBox)
        clover_web_logger.addHandler(ch)

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
        logging.info("start_app")
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

    def closeEvent(self, event):
        """Reimplemented to stop the notification handler before closing."""
        print("close_app")
        self._stop_event.set()
        self.handler.stop()  # Ensure the handler's thread is stopped
        tray_logger.debug("Stop event set")
        super().closeEvent(event)

    # Our custom slot that handles the tray icon activation
    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:  # Tray icon was clicked
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()  # Bring window to front


if __name__ == '__main__':
    # c_log.setLevel(logging.INFO)
    app = QApplication(sys.argv)
    window = TrayApp()
    window.show()
    sys.exit(app.exec_())
