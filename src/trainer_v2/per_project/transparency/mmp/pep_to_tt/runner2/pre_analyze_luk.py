import itertools
import random
import sys
import sys
from omegaconf import OmegaConf

from misc_lib import path_join, batch_iter_from_entry_iter
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_encoders import AlignCandidateExtractor


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    analyzer = Analyzer(get_lucene_analyzer(stemmer="krovetz"))
    extractor = AlignCandidateExtractor(analyzer.analyze)
    line_per_job = conf.line_per_job
    raw_train_iter = tsv_iter(conf.qd_triplet_file)
    random.seed(0)
    start_job = 0
    st = line_per_job * start_job
    itr = itertools.islice(raw_train_iter, st, None)
    itr = batch_iter_from_entry_iter(itr, line_per_job)
    for idx, batch in enumerate(itr):
        job_no = idx + start_job
        save_path = path_join(conf.candidate_dir, str(job_no) + ".txt")
        extractor.pre_analyze_print_per_query(batch, save_path)
        if idx > 40000:
            break

if __name__ == "__main__":
    main()
