from data_generator.bert.create_training_data import *
from data_generator.tokenizer_b import CLS_ID, SEP_ID
import pickle
import numpy as np
from misc_lib import TimeEstimator

mask_id = None

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def sample(probs, rng):
    val = rng.random()
    i = 0
    acc = 0
    while acc < val:
        acc += probs[i]
        i+= 1

        assert i < 10000

    assert i-1 < len(probs)
    return i-1

def new_sample_probs(probs, selected_index, tokens):
    indice = []
    new_probs = []

    for i in range(len(probs)):
        token = tokens[i]
        if not i in selected_index and token != CLS_ID and token != SEP_ID:
            new_probs.append(probs[i])
            indice.append(i)

    total = sum(new_probs)
    new_probs = list([v/total for v in new_probs])
    return new_probs, indice


def create_masked_lm_predictions(tokens, probs, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""
    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    soft_probs = softmax(np.array(probs))

    ## instead of shuffling, we select keywords
    cand_indexes = []
    cnt = 0
    while len(cand_indexes) < num_to_predict:
        cnt += 1
        if cnt > 10:
            new_probs, indice = new_sample_probs(soft_probs, cand_indexes, tokens)
            i_meta = sample(new_probs, rng)
            i = indice[i_meta]
        else:
            i = sample(soft_probs, rng)

        if i not in cand_indexes:
            token = tokens[i]
            if token == CLS_ID or token == SEP_ID:
                continue

            cand_indexes.append(i)
        if cnt > 999:
            print("len(tokens)", len(tokens))
            print(tokens)
            print(soft_probs)
            print(new_probs)
            print(cand_indexes)
            print(sum(new_probs))
            print(num_to_predict)
        assert cnt < 1000
    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = mask_id
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)



def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    SMALL_VALUE = -100000
    i = 0
    while i < len(document):
        segment = document[i]  # segment = (tokens, probs)
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                probs_a = []
                for j in range(a_end):
                    tokens, probs = current_chunk[j]
                    tokens_a.extend(tokens)
                    probs_a.extend(probs)

                tokens_b = []
                probs_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens, probs = random_document[j]
                        tokens_b.extend(tokens)
                        probs_b.extend(probs)
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens, probs = current_chunk[j]
                        probs_b.extend(probs)
                        tokens_b.extend(tokens)
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)
                truncate_seq_pair(probs_a, probs_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                probs = []
                segment_ids = []
                tokens.append(CLS_ID)
                segment_ids.append(0)

                assert len(probs_a) == len(tokens_a)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                for prob in probs_a:
                    probs.append(prob)

                probs.append(SMALL_VALUE)
                tokens.append(SEP_ID)
                segment_ids.append(0)

                assert len(probs_b) == len(tokens_b)
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                for prob in probs_b:
                    probs.append(prob)

                probs.append(SMALL_VALUE)
                tokens.append(SEP_ID)
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                         tokens, probs, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                        tokens=tokens,
                        segment_ids=segment_ids,
                        is_random_next=is_random_next,
                        masked_lm_positions=masked_lm_positions,
                        masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances



def create_training_instances(input_files, tokenizer, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob,
                                                            max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        obj = pickle.load(open(input_file, "rb"))
        # obj = List[Document]
        # Document = List[(Tokens, Probs)]
        all_documents.extend(obj)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.values())
    instances = []
    ticker = TimeEstimator(dupe_factor * len(all_documents))
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                    create_instances_from_document(
                            all_documents, document_index, max_seq_length, short_seq_prob,
                            masked_lm_prob, max_predictions_per_seq, vocab_words, rng))
            ticker.tick()
    rng.shuffle(instances)
    return instances


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = instance.tokens
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = instance.masked_lm_labels
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
          [tokenizer_b.printable_text(x) for x in tokenizer.convert_ids_to_tokens(instance.tokens)]))

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


def main(_):
    tf.logging.info("Targeted Pretraining")
    tf.logging.set_verbosity(tf.logging.INFO)


    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("    %s", input_file)
    tokenizer = tokenizer_b.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    global mask_id
    mask_id = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
            input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
            FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
            rng)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
