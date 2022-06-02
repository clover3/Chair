from typing import List, Set, Tuple, Dict

from trainer_v2.chair_logging import c_log


class NodeWrap:
    def __init__(self, token, t_idx):
        self.token = token
        self.t_idx = t_idx
        self.child_list: List[NodeWrap] = []
        self.subtree_size = None
        self.subtree_coverage = None

    def add_child(self, token):
        self.child_list.append(token)

    def is_root(self):
        return self.token.head == self.token

    def get_subtree_size(self) -> int:
        if self.subtree_size is not None:
            return self.subtree_size
        s = self._get_subtree_size()
        self.subtree_size = s
        return s

    def _get_subtree_size(self) -> int:
        s = 1
        for child in self.child_list:
            s += child.get_subtree_size()
        return s

    def get_subtree_coverage(self) -> Set:
        if self.subtree_coverage is not None:
            return self.subtree_coverage

        coverage = {self.t_idx}
        for c in self.child_list:
            coverage.update(c.get_subtree_coverage())

        self.subtree_coverage = coverage
        return coverage


class VirtualRootNode:
    def __init__(self, child_list):
        self.child_list: List[NodeWrap] = child_list
        self.subtree_size = None
        self.subtree_coverage = None

    def is_root(self):
        return True

    def get_subtree_size(self) -> int:
        if self.subtree_size is not None:
            return self.subtree_size
        s = self._get_subtree_size()
        self.subtree_size = s
        return s

    def _get_subtree_size(self) -> int:
        s = 1
        for child in self.child_list:
            s += child.get_subtree_size()
        return s

    def get_subtree_coverage(self) -> Set:
        if self.subtree_coverage is not None:
            return self.subtree_coverage

        coverage = set()
        for c in self.child_list:
            coverage.update(c.get_subtree_coverage())

        self.subtree_coverage = coverage
        return coverage


def consecutive(seq):
    for i in range(len(seq) - 1):
        if seq[i] + 1 != seq[i + 1]:
            return False
    return True


def split_spacy_tokens(spacy_tokens) -> Tuple[str, str]:
    node_list = get_tree_nodes(NodeWrap, spacy_tokens)
    root_list = [n for n in node_list if n.is_root()]
    if len(root_list) > 1:
        root = VirtualRootNode(root_list)
    elif len(root_list) == 0:
        c_log.warn("Number of root is {}".format(len(root_list)))
        raise ValueError
    else:
        root = root_list[0]
    # for t in spacy_tokens:
    #     print(t, t.head)
    graph_size = len(node_list)

    # return best node and it's score (lower the better)
    def search_node_score(node: NodeWrap) -> Tuple[NodeWrap, int]:
        subtree_size = node.get_subtree_size()
        remaining_size = graph_size - subtree_size
        score = abs(remaining_size - subtree_size)
        best_score = score
        best_node = node

        for child in node.child_list:
            child_best, child_best_score = search_node_score(child)
            if child_best_score < best_score:
                best_score = child_best_score
                best_node = child_best

        return best_node, best_score

    best_node, best_score = search_node_score(root)
    # print("Best node : ", best_node.token)
    group1_indices = best_node.get_subtree_coverage()
    group1_indices = list(group1_indices)
    group1_indices.sort()
    st = group1_indices[0]
    ed = group1_indices[-1] + 1
    if not consecutive(group1_indices):
        missing = 0
        for i in range(st, ed):
            if i not in group1_indices:
                missing += 1

        group1_indices = list(range(st, ed))
    group1_span = spacy_tokens[st:ed]

    group2_indices = [i for i in range(len(spacy_tokens)) if i not in group1_indices]

    group2_span_list = get_spans(spacy_tokens, group2_indices)
    mask_token = '[MASK]'

    def span_list_to_str(span_list, n_total_tokens):
        text_list = map(str, span_list)
        join_text = " " + mask_token + " "
        out_text = join_text.join(text_list)
        if not span_list:
            return ""

        first_span = span_list[0]
        if first_span.start != 0:
            out_text = mask_token + " " + out_text

        last_span = span_list[-1]
        if last_span.end < n_total_tokens:
            out_text = out_text + " " + mask_token
        return out_text

    group1_text = span_list_to_str([group1_span], len(spacy_tokens))
    group2_text = span_list_to_str(group2_span_list, len(spacy_tokens))

    assert len(group2_span_list) <= 2
    return group1_text, group2_text


def split_spacy_tokens_no_mask(spacy_tokens) -> Tuple[str, str]:
    node_list = get_tree_nodes(NodeWrap, spacy_tokens)
    root_list = [n for n in node_list if n.is_root()]
    if len(root_list) > 1:
        root = VirtualRootNode(root_list)
    elif len(root_list) == 0:
        c_log.warn("Number of root is {}".format(len(root_list)))
        raise ValueError
    else:
        root = root_list[0]
    # for t in spacy_tokens:
    #     print(t, t.head)
    graph_size = len(node_list)

    # return best node and it's score (lower the better)
    def search_node_score(node: NodeWrap) -> Tuple[NodeWrap, int]:
        subtree_size = node.get_subtree_size()
        remaining_size = graph_size - subtree_size
        score = abs(remaining_size - subtree_size)
        best_score = score
        best_node = node

        for child in node.child_list:
            child_best, child_best_score = search_node_score(child)
            if child_best_score < best_score:
                best_score = child_best_score
                best_node = child_best

        return best_node, best_score

    best_node, best_score = search_node_score(root)
    # print("Best node : ", best_node.token)
    group1_indices = best_node.get_subtree_coverage()
    group1_indices = list(group1_indices)
    group1_indices.sort()
    st = group1_indices[0]
    ed = group1_indices[-1] + 1
    if not consecutive(group1_indices):
        missing = 0
        for i in range(st, ed):
            if i not in group1_indices:
                missing += 1

        group1_indices = list(range(st, ed))
    group1_span = spacy_tokens[st:ed]

    group2_indices = [i for i in range(len(spacy_tokens)) if i not in group1_indices]
    group2_span_list = get_spans(spacy_tokens, group2_indices)

    def span_list_to_str(span_list):
        text_list = map(str, span_list)
        return " ".join(text_list)

    group1_text = span_list_to_str([group1_span])
    group2_text = span_list_to_str(group2_span_list)

    assert len(group2_span_list) <= 2
    return group1_text, group2_text



def get_spans(spacy_tokens, group2_indices):
    group2_span_list = []
    if not group2_indices:
        return []
    st = group2_indices[0]
    for i, idx in enumerate(group2_indices):
        next_idx = None
        try:
            next_idx = group2_indices[i + 1]
            is_next_consecutive = idx + 1 == next_idx
        except IndexError:
            is_next_consecutive = False

        if not is_next_consecutive:
            span = spacy_tokens[st:idx + 1]
            group2_span_list.append(span)
            st = next_idx
    return group2_span_list


def get_tree_nodes(NodeWrap, spacy_tokens) -> List[NodeWrap]:
    node_list = [NodeWrap(t, t_idx) for t_idx, t in enumerate(spacy_tokens)]
    idx_to_node: Dict[int, NodeWrap] = {}
    for node in node_list:
        idx_to_node[node.token.idx] = node

    def get_head_node(node: NodeWrap) -> NodeWrap:
        return idx_to_node[node.token.head.idx]

    for n in node_list:
        if not n.is_root():
            parent = get_head_node(n)
            parent.add_child(n)
    return node_list