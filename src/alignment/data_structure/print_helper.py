from bert_api.segmented_instance.segmented_text import seg_to_text
from alignment.data_structure.related_eval_instance import RelatedEvalInstance


def rei_to_text(tokenizer, rei: RelatedEvalInstance):
    seg1_text = seg_to_text(tokenizer, rei.seg_instance.text1)
    seg2_text = seg_to_text(tokenizer, rei.seg_instance.text2)
    return f"RelatedEvalInstance({rei.problem_id}, {seg1_text})\n" \
           + "Doc: " + seg2_text