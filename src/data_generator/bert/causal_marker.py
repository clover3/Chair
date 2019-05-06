from data_generator.bert.create_training_data import *
from misc_lib import pick1
import re
import os
import path
from text_lib import line_split

NO_MARKER = 0
CAUSAL_MARKER = 1
INV_MARKER = 2

def parse_causal_marker(tokens):
  single_markers = {
    'thus':1,
    'because':2
  }

  multiple_markers = [
    ['as', 'a','result']
  ]

  if len(tokens) == 0:
    return 0, tokens

  for pattern in single_markers.keys():
    if tokens[0] == pattern:
      return single_markers[pattern], tokens[1:]

  multi_marker_base = 3
  for idx, pattern in enumerate(multiple_markers):
    match = True
    for j, word in enumerate(pattern):
      if len(tokens) <= j or tokens[j] != word:
        match = False

    if match:
      pattern_id = idx + multi_marker_base
      return pattern_id, tokens[len(pattern):]

  # Not matched
  return 0, tokens


class CausalInstance(object):
  """A single training instance (sentence pair)."""
  def __init__(self, tokens, segment_ids, labels):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.labels = labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenizer_b.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "labels: %s\n" % str(self.labels)
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def expand_context(document, loc, target_seq_length):
  tokens, labels = document[loc]
  if len(tokens) > target_seq_length:
    tokens = tokens[:target_seq_length]

  seg_a = []
  seg_b = tokens
  current_length = len(tokens)
  a_cursor = loc - 1
  b_cursor = loc + 1
  while current_length < target_seq_length:
    if a_cursor >= 0:
      tokens = document[a_cursor][0]
      if current_length + len(tokens) <= target_seq_length:
        seg_a = tokens + seg_a
        current_length += len(tokens)
        a_cursor -= 1
        assert len(seg_a) + len(seg_b) <= target_seq_length
      else:
        break

    if b_cursor < len(document):
      tokens = document[b_cursor][0]
      if current_length + len(tokens) <= target_seq_length:
        seg_b.extend(tokens)
        current_length += len(tokens)
        b_cursor += 1
        assert len(seg_a) + len(seg_b) <= target_seq_length
      else:
        break

    if a_cursor < 0 and len(document) <= b_cursor:
      break
  assert len(seg_a) + len(seg_b) <= target_seq_length
  return seg_a, seg_b

def encode_seq(seg_a, seg_b):
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in seg_a:
    tokens.append(token)
    segment_ids.append(0)

  tokens.append("[SEP]")
  segment_ids.append(0)

  for token in seg_b:
    tokens.append(token)
    segment_ids.append(1)
  tokens.append("[SEP]")
  segment_ids.append(1)
  return tokens, segment_ids

def create_instances(document, max_seq_length, short_seq_prob, rng, marked_locs):
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

  instances = []
  for loc in marked_locs:
    if loc == 0:
      continue
    seg_a, seg_b = expand_context(document, loc, target_seq_length)

    if len(seg_a) == 0:
      continue
    _, labels = document[loc]

    tokens, segment_ids = encode_seq(seg_a, seg_b)
    instance = CausalInstance(
        tokens=tokens,
        segment_ids=segment_ids,
        labels=labels)
    instances.append(instance)
  return instances


def read_documents(input_files, tokenizer):
  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.

  all_documents = [[]]
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenizer_b.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])

        tokens = tokenizer.tokenize(line)
        label, tokens = parse_causal_marker(tokens)

        if tokens:
          all_documents[-1].append((tokens, label))
  return all_documents

def create_training_instances(docs, max_seq_length,
                              dupe_factor, short_seq_prob, rng):
  """Create `TrainingInstance`s from raw text."""

  # Remove empty documents
  docs = [x for x in docs if x]

  def mark_doc(doc):
    new_doc = []
    for line in doc:
      tokens = tokenizer.tokenize(line)
      label, tokens = parse_causal_marker(tokens)
      new_doc.append((tokens, label))
    return new_doc

  all_documents = map(mark_doc, docs)

  n_pos = 0
  n_neg = 0
  print_limit = 1000
  instances = []
  for _ in range(dupe_factor):
    for document in all_documents:
      pos_locs = list([i for i, d in enumerate(document) if d[1] > 0])
      if pos_locs:
        pos_insts = create_instances(document, max_seq_length, short_seq_prob, rng, pos_locs)
        n_pos += len(pos_insts)
        instances.extend(pos_insts)

      if n_neg < n_pos * 2:
        neg_locs = list([i for i, d in enumerate(document) if d[1] == 0])
        rng.shuffle(neg_locs)

        neg_insts = create_instances(document, max_seq_length, short_seq_prob, rng, neg_locs[:3])
        n_neg += len(neg_insts)
        instances.extend(neg_insts)

      if print_limit < n_pos:
        print("n_pos", n_pos)
        print("n_neg", n_neg)
        print_limit += 1000
  rng.shuffle(instances)
  return instances


def write_instance_to_example_files(instances, tokenizer, max_seq_length, output_files):
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
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label = instance.labels

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["label_ids"] = create_int_feature([label])

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


def parse_grep_result(filename):
  document = []
  cnt = 0
  with tf.gfile.GFile(filename, "r") as reader:
    while True:
      line = reader.readline()
      if not line:
        break
      line = line.strip()
      if line == "--":
        yield document
        cnt += 1
        if cnt > 1000:
          return
        document = []

      else:
        sentences = line_split(line)
        document.extend(sentences)


if __name__ == '__main__':
  causal_path = os.path.join(path.data_path,"causal")
  tf.logging.set_verbosity(tf.logging.INFO)
  src_file = os.path.join(causal_path,"Thus.txt")
  docs = parse_grep_result(src_file)
  rng = random.Random(FLAGS.random_seed)
  voca_path = os.path.join(path.data_path, "bert_voca.txt")

  tokenizer = tokenizer_b.FullTokenizer(
      vocab_file=voca_path, do_lower_case=True)

  max_seq_len = 256
  print("Creating Instances")
  insts = create_training_instances(docs, max_seq_len, 1, 0.1, rng)

  out_path_list = []
  for i in range(10):
    out_path = os.path.join(causal_path, "Thus.rm_{}".format(i))
    out_path_list.append(out_path)
  write_instance_to_example_files(insts, tokenizer, max_seq_len, out_path_list)