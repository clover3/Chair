from collections import Counter

from dataset_specific.mnli.snli_reader_tfds import SNLIReaderTFDS

reader = SNLIReaderTFDS()

counter = Counter()
for t in reader.get_train():
    l = len(t.premise.split())
    if l > 150:
        counter[150] += 1
    if l > 280:
        counter[280] += 1
    counter[0] += 1

print(counter)
