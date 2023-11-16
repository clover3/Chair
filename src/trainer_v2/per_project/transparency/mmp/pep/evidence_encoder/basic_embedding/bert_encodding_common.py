import pickle

from list_lib import left
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.basic_embedding.bert_encoder_wrap import \
    BERTEncoderWrap


def process_batch(
        job_no,
        return_key,
        item_per_job=10000,
        file_path='input.tsv',
        output_file='output.pkl'):
    # Read the file
    all_terms = left(tsv_iter(file_path))
    # Partition the data into batches and select the batch based on job_no
    start_index = job_no * item_per_job
    end_index = start_index + item_per_job
    words_batch = all_terms[start_index:end_index]

    for item in words_batch:
        if type(item) != str:
            print("item {} is not string: {}".format(item, type(item)))

    # Encode the words
    encoder = BERTEncoderWrap(return_key)
    embeddings = encoder.batch_encode(words_batch)

    # Save the embeddings to a pickle file
    with open(output_file, 'wb') as f:
        obj = words_batch, embeddings
        pickle.dump(obj, f)