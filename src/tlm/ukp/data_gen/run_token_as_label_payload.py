


# TODO load ukp text data:
# Encode topic as same segment as text
# Format: [CLS] [Abortion] [LABEL_FAVOR] ...(ukp text)...[SEP] [ABORTION] [LABEL_UNK] ..(clue text).. [SEP]
# TODO load clueweb
# [LABEL_FAVOR] is replaced with [LABEL_UNK] from data generation step
# add 'label_ids' feature to indicate label
# add 'label_loc' feature to indicate label location
# label_ids, label_loc -> [n, 1]
# model_fn : it randomly mask tokens, except special tokens,


import data_generator.argmining.ukp_header
from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.ukp.data_gen.token_as_label_payload_worker import UkpTokenLabelPayloadWorker, UkpTokenAsLabelGenerator

if __name__ == "__main__":
    num_jobs = len(data_generator.argmining.ukp_header.all_topics) - 1
    generator = UkpTokenAsLabelGenerator()
    JobRunner(sydney_working_dir, num_jobs, "ukp_cluweb_mix_payload",
              lambda x: UkpTokenLabelPayloadWorker(x, generator)).start()


