from omegaconf import OmegaConf

from adhoc.bm25_class import BM25Bare
from data_generator.tokenizer_wo_tf import get_tokenizer
from adhoc.other.bm25_retriever_helper import get_bm25_stats_from_conf
from trainer_v2.per_project.transparency.mmp.pep_to_tt.bm25_match_analyzer import BM25_MatchAnalyzer
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_common import get_pep_predictor
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_encoders import PEP_TT_EncoderSingle
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


def get_pep_tt_lucene_based_encoder(model_config: PEP_TT_ModelConfig, conf):
    bert_tokenizer = get_tokenizer()
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    avdl, cdf, df, dl_d = get_bm25_stats_from_conf(bm25_conf, None)
    bm25 = BM25Bare(df, len(dl_d), avdl, bm25_conf.k1, bm25_conf.k2, bm25_conf.b)

    from pyserini.analysis import Analyzer, get_lucene_analyzer
    analyzer = Analyzer(get_lucene_analyzer())

    get_pep_top_k = get_pep_predictor(conf)
    bm25_analyzer = BM25_MatchAnalyzer(bm25, get_pep_top_k, analyzer.analyze)
    # We will use pyserini's prebuit index.
    # It uses porter stemmer, but porter stemmer's output is not actual english words,
    # so it might be harmful for BERT.
    term_to_subword = bert_tokenizer.tokenize

    # Look up porter
    return PEP_TT_EncoderSingle(bert_tokenizer, model_config, bm25_analyzer, term_to_subword)



