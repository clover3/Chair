import time
from multiprocessing import Process, Queue


class QueueFeader:
    def __init__(self, max_elem, fn_gen_item, gen_list = False):
        self.queue = Queue(maxsize=max_elem)
        self.fn_gen_item = fn_gen_item
        self.hot = False
        self.gen_list = gen_list
        self.t = Process(target=self.worker)
        self.t.daemon = True
        self.t.start()

    def worker(self):
        print("QueueFeader Started")
        while True:
            batch = self.fn_gen_item()
            if self.queue.empty() :
                print("queue is empty")

            if self.gen_list:
                for item in batch:
                    self.queue.put(item, block=True)
            else:
                self.queue.put(batch, block=True)


    def get(self):
        ret = self.queue.get(True)
        return ret




