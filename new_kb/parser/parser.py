import os
import json

from tree_sitter import Language, Parser
from pathlib import Path
from new_kb.parser.parser_dicts import EXTENSION_TO_MODULE, FUNCTION_TYPES, CLASS_TYPES, IMPORT_TYPES, INTERFACE_TYPES, COMMENT_TYPES

# ----------------------------------------------------------------------
# Name extraction helpers (unchanged)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Import name extraction (unchanged)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Helpers for async and method detection
# ----------------------------------------------------------------------
def is_async_function(node):
    """
    Determine if a function node is async.
    Checks the node's type for 'async' prefix, or looks for an 'async' child token.
    """
    # Many grammars use a specific node type like async_function_definition
    if 'async' in node.type.lower():
        return True

    # For other languages, scan children for an 'async' token
    # (Tree-sitter often has a token with type 'async')
    for child in node.children:
        if child.type == 'async' or child.text.decode(errors="replace") == 'async':
            return True
    return False

def is_method(node, ancestors, class_types):
    """
    Returns True if the node is a function that is defined inside a class.
    """
    for anc in ancestors:
        if anc.type in class_types:
            return True
    return False

# ----------------------------------------------------------------------
# Main metric extraction (improved)
# ----------------------------------------------------------------------
def extract_file_metrics(root_node, extension, source_bytes):
    """
    Traverse the AST and build a comprehensive metrics dictionary.
    Returns:
      - counts for classes, functions (free), methods (inside classes),
        async_functions, imports (total count), lines, interfaces, comments
      - lists of function_names, class_names, and imported names
      - dictionaries of function/class definitions by name
    """
    source_text = source_bytes.decode(errors="replace")
    total_lines = len(source_text.splitlines())

    results = {
        "classes": 0,
        "functions": 0,          # top-level (free) functions
        "methods": 0,            # functions inside classes
        "async_functions": 0,
        "import_count": 0,
        "interface_count": 0,
        "comment_count": 0,
        "lines": total_lines,
        "function_names": [],
        "class_names": [],
        "function_definitions": {},
        "class_definitions": {},
        "imports": [],           # list of imported names (strings)
    }

    func_types = set(FUNCTION_TYPES.get(extension, []))
    class_types = set(CLASS_TYPES.get(extension, []))
    import_types = set(IMPORT_TYPES.get(extension, []))
    interface_types = set(INTERFACE_TYPES.get(extension, []))
    comment_types = set(COMMENT_TYPES.get(extension, []))

    def traverse(node, ancestors=None):
        if ancestors is None:
            ancestors = []

        # ---- Functions ----
        if node.type in func_types:
            is_async = is_async_function(node)
            is_method_flag = is_method(node, ancestors, class_types)

            if is_method_flag:
                results["methods"] += 1
            else:
                results["functions"] += 1

            if is_async:
                results["async_functions"] += 1

            # Extract name and store definition
            name = get_node_name(node, extension)
            if name:
                results["function_names"].append(name)
                results["function_definitions"][name] = node.text.decode(errors="replace")
            else:
                results["function_names"].append("<anonymous>")
                results["function_definitions"]["<anonymous>"] = node.text.decode(errors="replace")

        # ---- Classes ----
        if node.type in class_types:
            results["classes"] += 1
            name = get_node_name(node, extension)
            if name:
                results["class_names"].append(name)
                results["class_definitions"][name] = node.text.decode(errors="replace")
            else:
                results["class_names"].append("<anonymous>")
                results["class_definitions"]["<anonymous>"] = node.text.decode(errors="replace")

        # ---- Imports ----
        if node.type in import_types:
            results["import_count"] += 1
            imported = extract_import_names(node, extension)
            results["imports"].extend(imported)

        # ---- Interfaces & Comments ----
        if node.type in interface_types:
            results["interface_count"] += 1
        if node.type in comment_types:
            results["comment_count"] += 1

        # Recurse
        for child in node.children:
            traverse(child, ancestors + [node])

    traverse(root_node)
    return results

# ----------------------------------------------------------------------
# Database‑ready reshape
# ----------------------------------------------------------------------
def reshape_for_db(parser_metrics):
    """
    Convert parser output to the dict expected by initialize_db.save_file().
    """
    return {
        "classes": parser_metrics["classes"],
        "functions": parser_metrics["functions"],
        "methods": parser_metrics["methods"],
        "async_functions": parser_metrics["async_functions"],
        "imports": parser_metrics["import_count"],
        "lines": parser_metrics["lines"],
    }

# ----------------------------------------------------------------------
# Existing functions: get_parser, get_tree (unchanged)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Main: demonstrate usage on a directory
# ----------------------------------------------------------------------
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