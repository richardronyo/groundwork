import os
import json

from tree_sitter import Language, Parser
from pathlib import Path
from parser_dicts import EXTENSION_TO_MODULE, FUNCTION_TYPES, CLASS_TYPES, IMPORT_TYPES, INTERFACE_TYPES, COMMENT_TYPES

def extract_identifier_from_node(node):
    """Recursively find the first child node that is an identifier or name."""
    if node.type in {"identifier", "name"}:
        return node.text.decode(errors="replace")
    for child in node.children:
        result = extract_identifier_from_node(child)
        if result:
            return result
    return None

def get_node_name(node, extension):
    """
    Try multiple ways to extract the name of a function/class node.
    Returns the name as a string, or None if not found.
    """
    # 1. Try field 'name' (most common)
    name_node = node.child_by_field_name('name')
    if name_node:
        return name_node.text.decode(errors="replace")

    # 2. Try field 'identifier'
    name_node = node.child_by_field_name('identifier')
    if name_node:
        return name_node.text.decode(errors="replace")

    # 3. Special case for C/C++: function_definition has a 'declarator' field
    if extension in {'.c', '.h', '.cpp'}:
        declarator = node.child_by_field_name('declarator')
        if declarator:
            return extract_identifier_from_node(declarator)

    # 4. Fallback: recursively search for any identifier/name node
    return extract_identifier_from_node(node)

def extract_import_names(node, extension):
    """
    Extract imported module/package names from an import node.
    Returns a list of strings (e.g., ['os'], ['flask', 'pathlib']).
    """
    names = []
    # Try common field names that hold the module path
    for field in ['name', 'module', 'path', 'namespace']:
        field_node = node.child_by_field_name(field)
        if field_node:
            # For dotted names, take the full text
            names.append(field_node.text.decode(errors="replace").strip())
            return names

    # For languages like C# (using_directive) or Python (import statement),
    # recursively collect identifier/dotted_name nodes.
    def collect(n):
        if n.type in {'identifier', 'name', 'dotted_name', 'qualified_name', 'namespace_name'}:
            # If it's a dotted name, include the whole text
            if n.type in {'dotted_name', 'qualified_name', 'namespace_name'}:
                names.append(n.text.decode(errors="replace").strip())
            else:
                names.append(n.text.decode(errors="replace").strip())
        else:
            for child in n.children:
                collect(child)
    collect(node)

    # If we found nothing, fallback to the entire node text (stripped)
    if not names:
        names.append(node.text.decode(errors="replace").strip())

    return names

def extract_file_metrics(root_node, extension, source_bytes):
    """Traverse the AST and build a comprehensive metrics dictionary."""
    results = {
        "function_count": 0,
        "class_count": 0,
        "import_count": 0,
        "interface_count": 0,
        "comment_count": 0,
        "function_names": [],
        "class_names": [],
        "function_definitions": {},   # name -> source text
        "class_definitions": {},      # name -> source text (optional)
        "imports": [],                # list of imported names
    }

    func_types = set(FUNCTION_TYPES.get(extension, []))
    class_types = set(CLASS_TYPES.get(extension, []))
    import_types = set(IMPORT_TYPES.get(extension, []))
    interface_types = set(INTERFACE_TYPES.get(extension, []))
    comment_types = set(COMMENT_TYPES.get(extension, []))

    def traverse(node):
        # Count and record functions
        if node.type in func_types:
            results["function_count"] += 1
            name = get_node_name(node, extension)
            if name:
                results["function_names"].append(name)
                results["function_definitions"][name] = node.text.decode(errors="replace")
            else:
                # Anonymous function fallback
                results["function_names"].append("<anonymous>")
                results["function_definitions"]["<anonymous>"] = node.text.decode(errors="replace")

        # Count and record classes
        if node.type in class_types:
            results["class_count"] += 1
            name = get_node_name(node, extension)
            if name:
                results["class_names"].append(name)
                results["class_definitions"][name] = node.text.decode(errors="replace")
            else:
                results["class_names"].append("<anonymous>")
                results["class_definitions"]["<anonymous>"] = node.text.decode(errors="replace")

        # Count imports and extract imported names
        if node.type in import_types:
            results["import_count"] += 1
            imported = extract_import_names(node, extension)
            results["imports"].extend(imported)

        # Count interfaces and comments
        if node.type in interface_types:
            results["interface_count"] += 1
        if node.type in comment_types:
            results["comment_count"] += 1

        for child in node.children:
            traverse(child)

    traverse(root_node)
    return results

def get_parser(path: str) -> Parser:
    extension = '.' + path.split('.')[-1]
    if extension not in EXTENSION_TO_MODULE:
        raise ValueError(f"Unsupported file extension: {extension}")

    tree_sitter_module = EXTENSION_TO_MODULE[extension]

    if extension == ".ts":
        tree_sitter_language = Language(tree_sitter_module.language_typescript())
    elif extension == ".tsx":
        tree_sitter_language = Language(tree_sitter_module.language_tsx())
    elif extension == ".ml":
        tree_sitter_language = Language(tree_sitter_module.language_ocaml())
    elif extension == ".mli":
        tree_sitter_language = Language(tree_sitter_module.language_ocaml_interface())
    elif extension == ".php":
        tree_sitter_language = Language(tree_sitter_module.language_php())
    else:
        tree_sitter_language = Language(tree_sitter_module.language())

    return Parser(tree_sitter_language)

def get_tree(path: str, parser: Parser):
    file_bytes = Path(path).read_bytes()
    return parser.parse(file_bytes)

if __name__ == '__main__':
    metrics = dict()
    for file in os.listdir('parser_test'):
        file_path = f'parser_test/{file}'
        extension = '.' + file.split('.')[-1]

        parser = get_parser(file_path)
        tree = get_tree(file_path, parser)
        source_bytes = Path(file_path).read_bytes()

        metrics[file] = extract_file_metrics(tree.root_node, extension, source_bytes)

    with open(f'parser_test/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4) 