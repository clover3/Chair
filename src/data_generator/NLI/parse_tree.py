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

    def get_all_nodes(self, travel_again = False):
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



def test():
    from data_generator.NLI.nli import DataLoader
    data_loader = DataLoader(123, "bert_voca.txt", True)
    for s1, s2, bp1, bp2 in data_loader.get_train_infos():

        tree1 = binary_parse_to_tree(bp1)#.travel_print()
        tree2 = binary_parse_to_tree(bp2)



if __name__ == "__main__":
    test()
