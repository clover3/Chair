import os
import pickle

from data_generator.job_runner import sydney_working_dir
from misc_lib import BinHistogram


def function():
    save_path = os.path.join(sydney_working_dir, "RobustPredictTokens", "1")
    obj = pickle.load(open(save_path, "rb"))

    bin = BinHistogram(lambda x: int(x / 512) )
    for doc_id in obj:
        bin.add(len(obj[doc_id]))

    print(bin.counter)




if __name__ =="__main__":
    function()