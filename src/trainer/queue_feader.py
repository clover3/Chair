
from multiprocessing import Process, Queue


class QueueFeader:
    def __init__(self, max_elem, fn_gen_item):
        self.queue = Queue(maxsize=max_elem)
        self.fn_gen_item = fn_gen_item
        self.t = Process(target=self.worker)
        self.t.start()

    def worker(self):
        print("QueueFeader Started")
        while True:
            batch = self.fn_gen_item()
            self.queue.put(batch, block=True)

    def get(self):
        return self.queue.get(True)




