from typing import List, Tuple

from tlm.qtype.partial_relevance.complement_search_pckg.complement_candidate_gen_if import SegJoinPolicyIF, \
    PartialSegment

from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance, SegmentedText


class FuncContentSegJoinPolicy(SegJoinPolicyIF):
    def __init__(self):
        pass

    def join_tokens(self, si: SegmentedInstance, new_tokens: PartialSegment, preserve_seg_idx):
        if preserve_seg_idx == 0: # Preserve func_tokens
            if new_tokens.n_seg > 1:
                raise Exception("preserve_seg_idx == 0 and new_tokens.n_seg > 1 is not expected")
            return self.join_new_content(si, new_tokens.data)
        elif preserve_seg_idx == 1:
            if new_tokens.n_seg == 1:
                head = new_tokens.data
                tail = []
                head_tail: Tuple[List[int], List[int]] = (head, tail)
                return self.join_new_func(si, head_tail)
            elif new_tokens.n_seg == 2:
                return self.join_new_func(si, new_tokens.data)
            else:
                raise Exception("new_tokens.n_seg > 2 is not expected")
        else:
            raise ValueError()

    def join_new_content(self, si: SegmentedInstance, new_content_tokens: List[int]) -> SegmentedInstance:
        func_token_indices = si.text1.seg_token_indices[0]
        content_token_indices = si.text1.seg_token_indices[1]
        content_st = min(content_token_indices)
        head = [si.text1.tokens_ids[i] for i in func_token_indices if i < content_st]
        tail = [si.text1.tokens_ids[i] for i in func_token_indices if i >= content_st]

        new_seg1 = head + new_content_tokens + tail
        new_head_indices = [i for i in func_token_indices if i < content_st]
        new_body_indices = [i+content_st for i, _ in enumerate(new_content_tokens)]
        tail_st = content_st + len(new_content_tokens)
        new_tail_indices = [i + tail_st for i in func_token_indices if i >= content_st]
        new_text1_seg1_indices = [new_head_indices + new_tail_indices, new_body_indices]
        return SegmentedInstance(
            SegmentedText(new_seg1, new_text1_seg1_indices),
            SegmentedText(si.text2.tokens_ids, si.text2.seg_token_indices),
            si.score
        )

    def join_new_func(self, si: SegmentedInstance, new_func_tokens: Tuple[List[int], List[int]]) -> SegmentedInstance:
        head, tail = new_func_tokens
        content_token_indices = si.text1.seg_token_indices[1]
        body: List[int] = [si.text1.tokens_ids[i] for i in content_token_indices]
        content_st = len(head)
        new_head_indices = list(range(len(head)))
        tail_st = len(head) + len(body)
        new_tail_indices = [idx + tail_st for idx, _ in enumerate(tail)]

        new_seg1 = head + body + tail
        new_body_indices = [i + content_st for i, _ in enumerate(content_token_indices)]
        new_text1_seg1_indices = [new_head_indices + new_tail_indices, new_body_indices]
        return SegmentedInstance(
            SegmentedText(new_seg1, new_text1_seg1_indices),
            SegmentedText(si.text2.tokens_ids, si.text2.seg_token_indices),
            si.score
        )


