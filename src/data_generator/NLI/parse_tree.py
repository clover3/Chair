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

    def all_leave_idx(self):
        if self.is_inner():
            result = []
            for c in self.children:
                result += c.all_leave_idx()
            return result
        else:
            return [self.idx]

class Tree:
    def __init__(self, root):
        self.root = root
        self.all_nodes = None
        self.leaf_nodes = None

    def get_all_nodes(self, travel_again=False):
        if self.all_nodes is None or travel_again:

            def recurse(node):
                r = [node]
                for c in node.children:
                    r += recurse(c)
                return r

            self.all_nodes = recurse(self.root)

        return self.all_nodes

    def get_leaf_nodes(self):
        if self.leaf_nodes is None:
            self.leaf_nodes = []
            all_nodes = self.get_all_nodes()
            for node in all_nodes:
                if not node.is_inner():
                    self.leaf_nodes.append(node)
        return self.leaf_nodes

    def sample_any_node(self):
        return pick1(self.get_all_nodes())

    def sample_leaf_node(self):
        return pick1(self.get_leaf_nodes())

    def print_leaves(self):
        for node in self.get_all_nodes():
            if not node.is_inner():
                print("{}] {}".format(node.idx, node.data), end=" ")
        print()

    def leaf_tokens(self):
        return list([n.data for n in self.get_leaf_nodes()])

def str_to_subtree(elems):
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
    if "(" not in binary_parse_str:
        stack.append(str_to_subtree([stack.pop()]))

    assert len(stack) == 1
    return Tree(stack[0])





def test():
    from data_generator.NLI.nli import DataLoader
    data_loader = DataLoader(123, "bert_voca.txt", True)
    for s1, s2, bp1, bp2 in data_loader.get_train_infos():
        tree1 = binary_parse_to_tree(bp1)
        tree2 = binary_parse_to_tree(bp2)

        tree1.print_leaves()
        tree2.print_leaves()



if __name__ == "__main__":
    test()
