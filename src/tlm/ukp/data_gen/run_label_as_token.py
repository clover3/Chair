

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
from tlm.ukp.data_gen.ukp_clueweb_mix_worker import UkpCluewebMixWorker, UkpCluewebMixGenerator

if __name__ == "__main__":
    top_k = 150000
    blind_topic = "abortion"
    num_jobs = len(data_generator.argmining.ukp_header.all_topics) - 1
    generator = UkpCluewebMixGenerator()
    JobRunner(sydney_working_dir, num_jobs, "ukp_cluweb_mix",
              lambda x: UkpCluewebMixWorker(x, top_k, blind_topic, generator)).start()


