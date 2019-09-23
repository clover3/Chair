import math
from misc_lib import average, flatten
from cie import dependency
from collections import Counter, defaultdict
import random
import nltk.stem
from itertools import permutations
stemmer = nltk.stem.PorterStemmer()


def is_valid_NP_VP(sent):
    return NotImplemented


def max_core_scor(sent, core_dict):
    candidate = []
    for word in sent:
        if word in core_dict:
            candidate.append(core_dict[word])
    return max(candidate)

def average_likelihood(sent, fn_get_prob):
    log_p = 0
    for i, w in enumerate(sent):
        p_list = []
        for j, w2 in enumerate(sent):
            if i is not j:
                p = fn_get_prob(w,w2)
                p_list.append(p)

        avg_p = average(p_list)
        log_p += math.log(avg_p)
    return math.exp(log_p)


def sent_score(sent, core_dict, fn_get_prob):

    core_score = max_core_scor(sent, core_dict)
    coherence_score = average_likelihood(sent, fn_get_prob)

    syntactic_score = is_valid_NP_VP(sent)

    final_score = syntactic_score * core_score * coherence_score
    return final_score

# Condition1 : Weight should be larger than key edge
# Dependency is observed
def filter_edges(head, tail, edges):
    head = head.lower()
    tail = tail.lower()
    key_weight = 0
    for e in edges:
        h = e['head'].lower()
        t = e['tail'].lower()
        score = e['score']
        if head == h and tail == t:
            key_weight = score
        elif head == t and tail == h :
            key_weight = score
    print(edges)
    print(head,tail)
    assert(key_weight > 0)
    return list(filter(lambda x:x['score'] > key_weight, edges))

class ContextException(Exception):
    None

class Node:
    def __init__(self, name):
        self.name = name
        self.children_left = []
        self.children_right = []

    def add_child_left(self, node):
        self.children_left.append(node)

    def add_child_right(self, node):
        self.children_right.append(node)

    def contain_child(self, name):
        name_st = stemmer.stem(name)
        for child in self.children_left + self.children_right:
            if name_st == stemmer.stem(child.name):
                return True
        return False

    def children(self):
        return self.children_left + self.children_right


def expand_context(head, tail, tag, context_edges):
    def travel_add(node, edge):
        v1 = edge['head'].lower()
        v2 = edge['tail'].lower()
        if v1 == node.name.lower() and not node.contain_child(v2):
            if edge['direction'] == 'left':
                node.add_child_left(Node(v2))
            else:
                node.add_child_right(Node(v2))
            return True
        elif v2 == node.name.lower() and not node.contain_child(v1):
            if edge['direction'] == 'left':
                node.add_child_right(Node(v1))
            else:
                node.add_child_left(Node(v1))
            return True
        for child in node.children():
            if travel_add(child, edge):
                return True
        return False

    root = Node(head)
    child1 = Node(tail)
    retry = True
    context_edges.sort(key=lambda x:x['score'], reverse=True)
    remain_edges = context_edges
    names_in_tree = {root.name.lower(), child1.name.lower()}
    deps_in_tree = {tag}

    def duplicate(e):
        v1 = e['head']
        v2 = e['tail']
        tag = e['tag']
        voca_duplicate = v1.lower() in names_in_tree and v2.lower() in names_in_tree
        dep_duplicate = tag in deps_in_tree
        return voca_duplicate or dep_duplicate

    def update_dups(e):
        names_in_tree.add(e['head'].lower())
        names_in_tree.add(e['tail'].lower())
        deps_in_tree.add(e['tag'])


    print("Expanding tree")
    while retry :
        retry = False
        new_edges = []
        for e in remain_edges:
            if not duplicate(e) and (travel_add(root, e) or travel_add(child1, e)):
                update_dups(e)
                print("add {}".format(e))
                retry = True
            else:
                new_edges.append(e)
        remain_edges = new_edges

    # returns optimal ordering
    def best_order_by_count(nodes):
        names = list([node.name for node in nodes])
        print("Ordering : {}".format(names))
        n_tokens = len(nodes)

        candidates = []
        for ordering in permutations(range(n_tokens), n_tokens):
            score = 0
            for i in range(n_tokens):
                for j in range(i+1, n_tokens):
                    score += count(i, j)
            candidates.append((ordering, score))

        candidates.sort(key=lambda x:x[1], reverse=True)
        print(candidates[0])
        return candidates[0][0]

    print("Flatten tree")
    # input : node
    # output : List[string]
    def tree2seq(node):
        if not node.children():
            return [node.name]

        left = [tree2seq(c) for c in node.children_left]
        right = [tree2seq(c) for c in node.children_right]
        seq = flatten(left) + [node.name] + flatten(right)
        return seq

    head_ex = " ".join(tree2seq(root))
    tail_ex = " ".join(tree2seq(child1))
    return head_ex, tail_ex

def dep2statement(key, dependency_edges):
    tag, tail, head = key

    context_edges = dependency_edges
    if len(context_edges) > 100 :
        raise ContextException("Too much contexts")

    for e in context_edges:
        print(e)
    head_ex, tail_ex = expand_context(head, tail, tag, context_edges)

    if tag == "dobj":
        statement = "One should(n't) {} {}".format(head, tail)
        statement = "One should(n't) {} {}".format(head_ex, tail_ex)
    elif tag == "nsubj":
        statement = "{} should(n't) {}".format(tail, head)
        statement = "{} should(n't) {}".format(tail_ex, head_ex)
    elif tag == "nsubjpass":
        statement = "{} should(n't) be {}".format(tail, head)
    else:
        print(key)
        raise Exception("Invalid key tag")
    return statement

def appear_at_least(dependency_edges, k):
    def identity(edge):
        return " ".join(edge.values())

    count = Counter([identity(edge) for edge in dependency_edges])
    unique_ids = set()
    for edge in dependency_edges:
        id = identity(edge)
        if count[id] >= k and id not in unique_ids:
            unique_ids.add(id)
            yield edge


# generate a sentence
def generate(all_sents, bow_score):
    relation_counter = Counter()
    random.shuffle(all_sents)
    relation_all = []

    min_k = 1
    for sent in all_sents[:500]:
        relations = dependency.extract(sent)
        relations_all = dependency.extract_verbose(sent)
        for r in relations:
            relation_counter[r] += 1
        for r in relations_all:
            relation_all.append(r)
    frequent_edges = list(appear_at_least(relation_all, min_k))


    def get_score(w):
        w = w.lower()
        if w in bow_score:
            return bow_score[w]
        else:
            return 0

    relation_score = Counter()
    for key in relation_counter.keys():
        if relation_counter[key] >= min_k:
            tag, tail, head = key
            score = get_score(tail) * get_score(head)
            relation_score[key] = score

    for key, score in relation_score.most_common(15):
        tag, tail, head = key
        count = relation_counter[key]
        print(f" {tag}, {tail}, {head} : {score}\t{count} times")
    if len(relation_score) == 0:
        raise ContextException("Not enough relations")
    best_key, best_score = list(relation_score.most_common(1))[0]
    dependency_edges = []
    for r in frequent_edges:
        head = r['head']
        tail = r['tail']
        score = get_score(tail) * get_score(head)
        if score >= best_score * 0.6:
            r['score'] = score
            dependency_edges.append(r)

    return dep2statement(best_key, dependency_edges)