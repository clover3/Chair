

from plyer import notification

notification.notify(
    title='My App Notification',
    message='This is a notification message from My App!',
    app_name='My App',
    timeout=10  # the notification will stay for 10 seconds
)
