import ast
import os
import shutil

from misc_lib import get_first, group_by


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, file_path):
        self.file_path = file_path
        self.current_class = None  # Tracks the current class scope
        self.defined_classes = {}
        self.referenced_classes = set()
        self.defined_functions = {}
        self.called_functions = set()

    def visit_ClassDef(self, node):
        # Record the current class name when entering a class definition
        previous_class = self.current_class
        self.current_class = node.name
        self.defined_classes[node.name] = self.file_path
        self.generic_visit(node)
        # Restore the previous class context when leaving the class definition
        self.current_class = previous_class

    def visit_FunctionDef(self, node):
        if self.current_class is None:
            # If not inside a class, record the function as a global function
            self.defined_functions[node.name] = self.file_path
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.called_functions.add(node.func.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        self.referenced_classes.add(node.id)
        self.generic_visit(node)


def analyze_files(file_paths):
    all_defined_classes = {}
    all_referenced_classes = set()
    all_defined_functions = {}
    all_called_functions = set()

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            tree = ast.parse(file.read())
            analyzer = CodeAnalyzer(file_path)
            analyzer.visit(tree)
            all_defined_classes.update(analyzer.defined_classes)
            all_referenced_classes.update(analyzer.referenced_classes)
            all_defined_functions.update(analyzer.defined_functions)
            all_called_functions.update(analyzer.called_functions)

    # Classes that are defined but not referenced
    unused_classes = [(path, cls) for cls, path in all_defined_classes.items() if cls not in all_referenced_classes]

    # Functions that are defined but not called
    unused_functions = [(path, func) for func, path in all_defined_functions.items() if func not in all_called_functions]

    return unused_classes, unused_functions


def list_python_files(directory):
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

# Example usage


def remove_unused_functions(file_paths, unused_functions):
    class CodeRewriter(ast.NodeVisitor):
        def __init__(self, file_path, unused_functions):
            self.file_path = file_path
            self.unused_functions = unused_functions
            self.lines_to_remove = set()

        def visit_FunctionDef(self, node):
            func_name = node.name
            if func_name in self.unused_functions:
                for lineno in range(node.lineno, node.end_lineno + 1):
                    self.lines_to_remove.add(lineno)
            self.generic_visit(node)

    for src_file_path in file_paths:
        with open(src_file_path, 'r') as file:
            content = file.readlines()
            tree = ast.parse(''.join(content), type_comments=True)

        rewriter = CodeRewriter(src_file_path, unused_functions)
        rewriter.visit(tree)
        shutil.copy(src_file_path, src_file_path + ".old")
        with open(src_file_path, 'w') as file:
            for i, line in enumerate(content, start=1):
                if i not in rewriter.lines_to_remove:
                    file.write(line)


def remove_unused_classes(file_paths, unused_classes):
    class CodeRewriter(ast.NodeVisitor):
        def __init__(self, file_path, unused_classes):
            self.file_path = file_path
            self.unused_classes = unused_classes
            self.lines_to_remove = set()

        def visit_ClassDef(self, node):

            if node.name in self.unused_classes:
                for lineno in range(node.lineno, node.end_lineno + 1):
                    self.lines_to_remove.add(lineno)
            self.generic_visit(node)


    for file_path in file_paths:
        with open(file_path, 'r') as file:
            content = file.readlines()
            tree = ast.parse(''.join(content), type_comments=True)

        rewriter = CodeRewriter(file_path, unused_classes)
        rewriter.visit(tree)
        shutil.copy(file_path, file_path + ".old")

        with open(file_path, 'w') as file:
            for i, line in enumerate(content, start=1):
                if i not in rewriter.lines_to_remove:
                    file.write(line)
        #

def main():
    target_dir = r"C:\work\code\cond-nli_work\src"
    py_files = list_python_files(target_dir)
    # Example usage
    unused_classes, unused_functions = analyze_files(py_files)

    for path, entries in group_by(unused_functions, get_first).items():
        print(path)
        func_list = []
        for _path, func in entries:
            print(f"Function '{func}' is not called")
            func_list.append(func)

    #
    # for path, entries in group_by(unused_classes, get_first).items():
    #     print(path)
    #     class_del_list = []
    #     for _path, class_i in entries:
    #         print(f"Class '{class_i}' is not called")
    #         class_del_list.append(class_i)


if __name__ == "__main__":
    main()