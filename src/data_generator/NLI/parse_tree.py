from misc_lib import pick1


class Node(object):
    def __init__(self, idx, data):
        self.idx = idx
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

    def travel_print(self):
        print(self.idx, self.data)
        for c in self.children:
            c.travel_print()

    def is_inner(self):
        return self.idx == -1


class Tree:
    def __init__(self):
        self.root = NotImplemented
        self.all_nodes = None

    def get_all_nodes(self, travel_again=False):
        if self.all_nodes is None or travel_again:

            def recurse(node):
                r = [node]
                for c in node.children:
                    r += recurse(c)
                return r

            self.all_nodes = recurse(self.root)

        return self.all_nodes

    def sample_any_node(self):
        return pick1(self.get_all_nodes())


def str_to_subtree(elems):
    assert len(elems) > 1  # assume there is no case that only one elem in the subtree

    parent = Node(-1, "inner")
    for elem in elems:
        if type(elem) == Node:
            parent.add_child(elem)
        elif len(elem) == 2 and type(elem[1]) == str:
            idx, data = elem
            parent.add_child(Node(idx, data))
        else:
            assert False
    return parent


def binary_parse_to_tree(binary_parse_str):
    stack = []
    tokens = binary_parse_str.split()
    idx = 0
    for token in tokens:
        if token == "(":
            stack.append(token)
        elif token == ")":
            new_node = []
            while stack[-1] != "(":
                elem = stack.pop()
                new_node.append(elem)
            stack.pop() # throw away "("
            new_node.reverse()
            stack.append(str_to_subtree(new_node))
        else:

            stack.append((idx,token))
            idx += 1

    assert len(stack) == 1
    return stack[0]


# Output : Find indice of subword_tokens that covers indice of parse_tokens
# Lowercase must be processed
# indice is for parse_tokens
def translate_index(parse_tokens, subword_tokens, indice):
    sep_char = "#"
    result = []

    for target_index in indice:
        prev_text = "".join(parse_tokens[:target_index])
        pt_idx = 0
        print("Target_index", target_index)
        print("prev_text : " + prev_text)

        # find the index in subword_tokens that covers
        f_inside = False
        for t_idx, token in enumerate(subword_tokens):
            for c in token:
                # Here, previous char should equal prev_text[text_idx]
                if c in [sep_char, " "]:
                    continue

                if c == prev_text[pt_idx]:
                    pt_idx += 1
                else:
                    assert False
                # Here, c should equal prev_text[text_idx-1]
                assert c == prev_text[pt_idx - 1]
            if f_inside:
                print("Keep adding : " + token)
                result.append(t_idx)
                if pt_idx == len(prev_text):
                    break
            else:
                if pt_idx == len(prev_text):
                    print("Add begin : " + token)
                    result.append(t_idx)
                    prev_text += parse_tokens[target_index]
                    f_inside = True
    return result

def test():
    from data_generator.NLI.nli import DataLoader
    data_loader = DataLoader(123, "bert_voca.txt", True)
    for s1, s2, bp1, bp2 in data_loader.get_train_infos():
        tree1 = binary_parse_to_tree(bp1)
        tree2 = binary_parse_to_tree(bp2)



if __name__ == "__main__":
    test()
