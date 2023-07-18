from cache import load_pickle_from
from list_lib import assert_length_equal
from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import load_str_float_tsv
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.parse_helper import read_qid_pid_score_tsv
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import IRLProxyIF, \
    IndexedRankedList


# IRL: IndexedRankedList
# The cache exploits the fact (q_term, d_term) is sorted so that the same q_terms
# are repeated and it will access the same qids
class IRLProxy2(IRLProxyIF):
    def __init__(
            self,
            shallow_score_dir,
            deep_score_dir,
            tfs_save_dir,
            profile,
            disable_cache=False,
    ):
        self.q_term = ""
        self.cached = {}
        self.profile = profile
        self.shallow_score_dir = shallow_score_dir
        self.deep_score_dir = deep_score_dir
        self.tfs_save_dir = tfs_save_dir
        self.disable_cache = disable_cache

    def read_shallow_score(self, qid):
        save_path = path_join(self.shallow_score_dir, qid)
        return load_str_float_tsv(qid, save_path)

    def read_deep_score(self, qid):
        save_path = path_join(self.deep_score_dir, qid)
        return read_qid_pid_score_tsv(qid, save_path)

    def set_new_q_term(self, q_term):
        if q_term == self.q_term or self.disable_cache:
            pass
        else:
            c_log.info("Reset cache ({}->{})".format(self.q_term, q_term))
            self.q_term = q_term
            self.cached = {}

    def get_irl(self, qid) -> IndexedRankedList:
        if qid in self.cached:
            return self.cached[qid]

        # c_log.debug("Loading for qid=%s", qid)
        tfs_save_path = path_join(self.tfs_save_dir, qid)
        qid_, pid_tfs = load_pickle_from(tfs_save_path)
        qid_s, pid_scores_s = self.read_shallow_score(qid)
        qid_d, pid_scores_d = self.read_deep_score(qid)

        assert qid_ == qid_s
        assert qid_ == qid_d
        assert_length_equal(pid_tfs, pid_scores_s)
        assert_length_equal(pid_tfs, pid_scores_d)

        e_out_list = []
        for i in range(len(pid_tfs)):
            pid_, tfs = pid_tfs[i]
            pid_s, shallow_model_score_base = pid_scores_s[i]
            pid_d, deep_model_score = pid_scores_d[i]
            assert pid_ == pid_d
            e_out = IndexedRankedList.Entry(
                doc_id=pid_,
                deep_model_score=deep_model_score,
                shallow_model_score_base=shallow_model_score_base,
                tfs=tfs
            )
            e_out_list.append(e_out)

        irl = IndexedRankedList(qid, e_out_list)
        if not self.disable_cache:
            self.cached[qid] = irl
        return irl