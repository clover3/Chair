import signal
import time


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.sig_int_handler)
        signal.signal(signal.SIGTERM, self.sig_term_handler)
        signal.signal(signal.SIGCONT, self.sig_cont_handler)

    def sig_term_handler(self, *args):
        print("sig_term_handler")
        self.exit_gracefully(args)

    def sig_cont_handler(self, *args):
        print("sig_cont_handler")

    def sig_kill_handler(self, *args):
        print("sig_kill_handler")
        self.exit_gracefully(args)

    def sig_int_handler(self, *args):
        print("sig_int_handler")
        self.exit_gracefully(args)

    def exit_gracefully(self, *args):
        print("exit_gracefully")
        self.kill_now = True


if __name__ == '__main__':
    killer = GracefulKiller()
    while not killer.kill_now:
        time.sleep(1)
        print("doing something in a loop ...")

    print("End of the program. I was killed gracefully :)")
