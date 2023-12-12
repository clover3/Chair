
import ast
import os
import shutil
from cpath import src_path
from misc_lib import make_parent_exists


def find_imports(file_path, base_path='', visited=None):
    if visited is None:
        visited = set()

    if file_path in visited:
        return set()
    visited.add(file_path)

    abs_path = os.path.join(base_path, file_path)
    with open(abs_path, 'r') as file:
        node = ast.parse(file.read())

    imports = set()
    for item in node.body:
        if isinstance(item, ast.Import):
            for alias in item.names:
                imports.add(alias.name)
        elif isinstance(item, ast.ImportFrom):
            imports.add(item.module)

    # Find local imports and check them recursively
    local_imports = set()
    for imp in imports:
        local_path = os.path.join(base_path, imp.replace('.', '/') + '.py')
        if os.path.exists(local_path):
            local_imports.add(imp)
            local_imports.update(find_imports(local_path, base_path, visited))

    return local_imports


def main():
    base_path = src_path
    # Example usage
    target_root = "C:\work\code\cond-nli_work\src"
    file_path = 'contradiction/medical_claims/token_tagging/nlits_runner/concat6_c.py'
    # file_path = "trainer_v2/custom_loop/runner/concat/mat.py"
    local_imports = find_imports(file_path, base_path)
    local_imports = list(local_imports)
    local_imports.sort()
    for item in local_imports:
        local_path = os.path.join(base_path, item.replace('.', '/') + '.py')
        dest_path = os.path.join(target_root, item.replace('.', '/') + '.py')
        make_parent_exists(dest_path)
        shutil.copyfile(local_path, dest_path)


if __name__ == "__main__":
    main()
