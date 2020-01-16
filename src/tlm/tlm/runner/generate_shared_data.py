import collections
import os
import random

from cpath import output_path
from data_generator.job_runner import sydney_working_dir, JobRunner
from misc_lib import get_dir_files
from tf_util.enum_features import load_record_v2
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.dictionary.feature_to_text import take


def get_dir_all_itr(dir_path):
    for file_path in get_dir_files(dir_path):
        one_itr = load_record_v2(file_path)
        for item in one_itr:
            yield item


def encode_runner(lm_data_path, itr_nli):
    itr_lm = get_dir_all_itr(lm_data_path)

    try :
        idx = 0
        while True:
            out_path = os.path.join(sydney_working_dir, "lm_nli", str(idx))
            idx += 1
            encode2(itr_lm, itr_nli, out_path)
    except StopIteration as e:
        print("End of itr")
        pass
    except Exception as e:
        print(e)
        pass


def encode2(itr_lm, itr_nli, out_path):
    writer = RecordWriterWrap(out_path)
    for nli_entry in itr_nli:
        lm_entry = itr_lm.__next__()
        new_features = combine_feature(lm_entry, nli_entry)
        writer.write_feature(new_features)
    print("Wrote {} items".format(writer.total_written))
    writer.close()


def combine_feature(lm_entry, nli_entry):
    new_features = collections.OrderedDict()
    for key in lm_entry:
        new_features[key] = create_int_feature(take(lm_entry[key]))
    for key in nli_entry:
        if key == "label_ids":
            new_features[key] = create_int_feature(take(nli_entry[key]))
        else:
            new_key = "nli_" + key
            new_features[new_key] = create_int_feature(take(nli_entry[key]))
    return new_features


class Worker:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.lm_dir = os.path.join(sydney_working_dir, "unmasked_pair_x3")
        tt_path = os.path.join(output_path, "ukp_512", "train_death_penalty")
        self.tt_entries = list(load_record_v2(tt_path))

    def work(self, job_id):
        file_path = os.path.join(self.lm_dir, str(job_id))
        out_path = os.path.join(self.working_dir, str(job_id))
        lm_itr = load_record_v2(file_path)
        random.shuffle(self.tt_entries)
        idx = 0
        writer = RecordWriterWrap(out_path)
        for lm_entry in lm_itr:
            nli_entry = self.tt_entries[idx]
            new_features = combine_feature(lm_entry, nli_entry)
            writer.write_feature(new_features)


if __name__ == "__main__":
    runner = JobRunner(sydney_working_dir, 1000, "ukp_lm", Worker)
    runner.start()


