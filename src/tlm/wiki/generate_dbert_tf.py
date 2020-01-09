import sys
import time

from cache import *
from cpath import data_path
from data_generator import tokenizer_wo_tf as tokenization
from sydney_manager import MarkedTaskManager
from tlm.wiki import bert_training_data as btd
from tlm.wiki.bert_training_data import *

working_path ="/mnt/nfs/work3/youngwookim/data/dbert_tf"

def add_d_info(instances, dictionary):
    def reconstruct(tokens):
        groups = [] 
        for idx, t in tokens:
            if t.startswith("##"):
                groups[-1].append((idx, t))
            else:
                groups.append([(idx, t)])
        new_group = [] 
        for e in groups:
             subtoken_list = []
             indice = []
             for idx, t in e:
                 subtoken_list.append(t)
                 indice.append(idx)
             new_group.append((subtoken_list, indice))
        return new_group
    def join(subwords):
        out_s = subwords[0]
        for s in subwords[1:]:
            assert "##" == s[:2]
            out_s += s[2:]
        return out_s

    def get_score(word):
        if word not in dictionary:
            return 0
        num_def = dictionary.get_n_def(word)
        idf = dictionary.idf(word)
        return idf / num_def

    for inst in instances:
        m = reconstruct(inst.tokens)

        candidates = []
        for e in m:
            subwords, indice = e
            skip = False
            for idx in indice:
                if idx in inst.masked_lm_positions:
                    skip = True
            if not skip:
                word = join(subwords)
                score = get_score(word)
                candidates.append((word, indice, score))

        candidates.sort(key=lambda x:x[2], reverse=True)

def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)

        d_input_ids = tokenizer.convert_tokens_to_ids(instance.d_tokens)
        d_input_mask = [1] * len(d_input_ids)
        d_loc_ids = [instance.d_loc] * len(d_input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["d_input_ids"] = create_int_feature(d_input_ids)
        features["d_input_mask"] = create_int_feature(d_input_mask)
        features["d_loc_ids"] = create_int_feature(d_loc_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                    [tokenizer_b.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                        "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)

def parse_wiki(file_path):
    f = open(file_path, "r")
    documents = []
    doc = list()
    for line in f:
        if line.strip():
            doc.append(line)
        else:
            documents.append(doc)
            doc = list()
    return documents


class Worker:
    def __init__(self, out_path):
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

        self.masked_lm_prob = 0.15
        self.short_seq_prob = 0.1
        self.problem_per_job = 100 * 1000
        self.max_seq_length = 512
        self.max_predictions_per_seq = 20
        self.dupe_factor = 1
        self.out_dir = out_path

        seed = time.time()
        self.rng = random.Random(seed)
        print("Loading documents")
        self.documents = self.load_documents_from_pickle()
        print("Loading documents Done : ", len(self.documents))

    def load_documents_from_pickle(self):
        seg_id = self.rng.randint(0, 9)
        file_path = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/tokens/enwiki_train_tokens.{}"
        all_docs = []
        for j in range(100):
            full_id = seg_id * 100 + j
            f = open(file_path.format(full_id), "rb")
            all_docs.extend(pickle.load(f))
        return all_docs





    def load_documents(self):
        i = self.rng.randint(0,9)
        file_path = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_train.txt.line.{}".format(i)
        print(file_path)
        docs = parse_wiki(file_path)

        out_docs = []
        # Empty lines are used as document delimiters
        ticker = TimeEstimator(len(docs))
        for doc in docs:
            out_docs.append([])
            for line in doc:
                line = line.strip()
                tokens = self.tokenizer.tokenize(line)
                if tokens:
                    out_docs[-1].append(tokens)


            ticker.tick()
        assert out_docs[3]
        return out_docs


    def work(self, job_id):
        output_file = os.path.join(self.out_dir, "{}".format(job_id))
        instances = btd.create_training_instances(
            self.documents, self.tokenizer, self.max_seq_length, self.dupe_factor,
            self.short_seq_prob, self.masked_lm_prob, self.max_predictions_per_seq,
            self.rng)
        instances = add_d_info(instances)
        write_instance_to_example_files(instances, self.tokenizer, self.max_seq_length,
                                        self.max_predictions_per_seq, [output_file])


def main():
    mark_path = os.path.join(working_path, "wiki_p2_mark")
    out_path = os.path.join(working_path, "tf")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    mtm = MarkedTaskManager(1, mark_path, 1)
    worker = Worker(out_path)

    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)

def simple():
    out_path = os.path.join(working_path, "tf")
    worker = Worker(out_path)

    worker.work(int(sys.argv[1]))


if __name__ == "__main__":
    main()

