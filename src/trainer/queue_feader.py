import time
from multiprocessing import Process, Queue


class QueueFeader:
    def __init__(self, max_elem, fn_gen_item):
        self.queue = Queue(maxsize=max_elem)
        self.fn_gen_item = fn_gen_item
        self.t = Process(target=self.worker)
        self.t.daemon = True
        self.t.start()
        self.hot = False

    def worker(self):
        print("QueueFeader Started")
        while True:
            batch = self.fn_gen_item()
            if self.queue.empty() :
                print("queue is empty")
            self.queue.put(batch, block=True)


    def get(self):
        ret = self.queue.get(True)
        return ret




