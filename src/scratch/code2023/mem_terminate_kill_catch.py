
import signal
import time

class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        print("exit_gracefully")
        self.kill_now = True


def main():
    killer = GracefulKiller()

    arr = []
    for _ in range(1000):
        print("current arr has {} items".format(len(arr)))
        arr.append([0] * int(1e9))

        if killer.kill_now:
            print("Terminate now")

    print("I terminate gracefully")

if __name__ == "__main__":
    main()