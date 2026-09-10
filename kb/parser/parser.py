import os
import json

from tree_sitter import Language, Parser
from pathlib import Path
from kb.parser.parser_dicts import EXTENSION_TO_MODULE, FUNCTION_TYPES, CLASS_TYPES, IMPORT_TYPES, INTERFACE_TYPES, COMMENT_TYPES

# ----------------------------------------------------------------------
# Name extraction (unchanged)
# ----------------------------------------------------------------------
def extract_identifier_from_node(node):
    if node.type in {"identifier", "name"}:
        return node.text.decode(errors="replace")
    for child in node.children:
        result = extract_identifier_from_node(child)
        if result:
            return result
    return None

def get_node_name(node, extension):
    name_node = node.child_by_field_name('name')
    if name_node:
        return name_node.text.decode(errors="replace")
    name_node = node.child_by_field_name('identifier')
    if name_node:
        return name_node.text.decode(errors="replace")
    if extension in {'.c', '.h', '.cpp'}:
        declarator = node.child_by_field_name('declarator')
        if declarator:
            return extract_identifier_from_node(declarator)
    return extract_identifier_from_node(node)

# ----------------------------------------------------------------------
# NEW: Structured extraction helpers
# ----------------------------------------------------------------------
def extract_base_classes(node, extension):
    """Return list of base class names (strings)."""
    bases = []
    # Python: 'superclasses' field
    superclasses = node.child_by_field_name('superclasses')
    if superclasses:
        for child in superclasses.children:
            if child.type in {'identifier', 'name', 'attribute', 'subscript'}:
                bases.append(child.text.decode(errors="replace").strip())
        return bases

    # JS/TS/C#: look for 'extends' or 'implements' clauses
    for child in node.children:
        if child.type in {'heritage', 'heritage_clause', 'base_list', 'extends_clause'}:
            for sub in child.children:
                if sub.type in {'identifier', 'name', 'type_identifier', 'nested_identifier'}:
                    bases.append(sub.text.decode(errors="replace").strip())
    return bases

def extract_attributes(node, extension):
    """Return list of dicts: {name, type, visibility} for class attributes."""
    attrs = []
    body = node.child_by_field_name('body')
    if not body:
        return attrs

    for child in body.children:
        # Python: assignment at class level
        if extension == '.py' and child.type == 'assignment':
            left = child.child_by_field_name('left')
            if left and left.type == 'identifier':
                name = left.text.decode(errors="replace")
                # type annotation?
                type_node = child.child_by_field_name('type')
                typ = type_node.text.decode(errors="replace") if type_node else None
                vis = 'private' if name.startswith('_') else 'public'
                attrs.append({'name': name, 'type': typ, 'visibility': vis})

        # JS/TS/C#: property definition
        elif child.type in {'property_definition', 'public_field_definition', 'field_definition'}:
            name = get_node_name(child, extension)
            if name:
                typ_node = child.child_by_field_name('type')
                typ = typ_node.text.decode(errors="replace") if typ_node else None
                vis = 'public'
                if child.type == 'public_field_definition':
                    vis = 'public'
                elif child.type == 'private_field_definition':
                    vis = 'private'
                elif child.type == 'protected_field_definition':
                    vis = 'protected'
                # Check for visibility modifiers in children
                for sub in child.children:
                    if sub.type in {'public', 'private', 'protected'}:
                        vis = sub.type
                attrs.append({'name': name, 'type': typ, 'visibility': vis})

        # Generic fallback: variable_declarator inside class
        elif child.type == 'variable_declaration':
            for dec in child.children:
                if dec.type == 'variable_declarator':
                    name = get_node_name(dec, extension)
                    if name:
                        typ = None  # hard to infer without type inference
                        attrs.append({'name': name, 'type': typ, 'visibility': 'public'})
    return attrs

def extract_parameters(node, extension):
    """Return list of parameter strings, e.g. ['a int', 'b str']."""
    params = []
    param_node = node.child_by_field_name('parameters')
    if not param_node:
        return params

    for child in param_node.children:
        if child.type in {'parameter', 'formal_parameter', 'required_parameter', 'optional_parameter'}:
            name = get_node_name(child, extension)
            typ_node = child.child_by_field_name('type')
            typ = typ_node.text.decode(errors="replace") if typ_node else None
            if name:
                params.append(f"{name} {typ}" if typ else name)
    return params

def extract_return_type(node, extension):
    """Return string or None."""
    ret = node.child_by_field_name('return_type')
    if ret:
        return ret.text.decode(errors="replace").strip()
    # Fallback: look for '->' or ':' token in some grammars
    for child in node.children:
        if child.type == 'type' and child.parent == node:
            return child.text.decode(errors="replace").strip()
    return None

def extract_visibility(node, extension):
    """Return 'public', 'private', 'protected', or None."""
    # Explicit keywords
    for child in node.children:
        if child.type in {'public', 'private', 'protected'}:
            return child.type
    # Python convention: leading underscore
    name = get_node_name(node, extension)
    if name and extension == '.py':
        if name.startswith('__'):
            return 'private'
        if name.startswith('_'):
            return 'protected'  # or 'private' depending on convention
        return 'public'
    # Default for languages without explicit visibility
    return 'public' if name else None

# ----------------------------------------------------------------------
# Import extraction (unchanged)
# ----------------------------------------------------------------------
def extract_import_names(node, extension):
    names = []
    for field in ['name', 'module', 'path', 'namespace']:
        field_node = node.child_by_field_name(field)
        if field_node:
            names.append(field_node.text.decode(errors="replace").strip())
            return names

    def collect(n):
        if n.type in {'identifier', 'name', 'dotted_name', 'qualified_name', 'namespace_name'}:
            names.append(n.text.decode(errors="replace").strip())
        else:
            for child in n.children:
                collect(child)
    collect(node)
    if not names:
        names.append(node.text.decode(errors="replace").strip())
    return names

# ----------------------------------------------------------------------
# Async / method detection (unchanged)
# ----------------------------------------------------------------------
def is_async_function(node):
    if 'async' in node.type.lower():
        return True
    for child in node.children:
        if child.type == 'async' or child.text.decode(errors="replace") == 'async':
            return True
    return False

def is_method(node, ancestors, class_types):
    for anc in ancestors:
        if anc.type in class_types:
            return True
    return False

# ----------------------------------------------------------------------
# Main metric extraction — NOW with detailed structures
# ----------------------------------------------------------------------
def extract_file_metrics(root_node, extension, source_bytes):
    source_text = source_bytes.decode(errors="replace")
    total_lines = len(source_text.splitlines())

    results = {
        "classes": 0,
        "functions": 0,
        "methods": 0,
        "async_functions": 0,
        "import_count": 0,
        "interface_count": 0,
        "comment_count": 0,
        "lines": total_lines,
        "imports": [],
        # NEW: detailed records for database insertion
        "class_details": [],     # list of dicts: {name, bases, attributes, ...}
        "function_details": [],  # list of dicts: {name, params, return_type, is_async, visibility, class_id?}
    }

    func_types = set(FUNCTION_TYPES.get(extension, []))
    class_types = set(CLASS_TYPES.get(extension, []))
    import_types = set(IMPORT_TYPES.get(extension, []))
    interface_types = set(INTERFACE_TYPES.get(extension, []))
    comment_types = set(COMMENT_TYPES.get(extension, []))

    def traverse(node, ancestors=None):
        if ancestors is None:
            ancestors = []

        # ---- Classes ----
        if node.type in class_types:
            name = get_node_name(node, extension) or "<anonymous>"
            bases = extract_base_classes(node, extension)
            attributes = extract_attributes(node, extension)

            results["classes"] += 1
            results["class_details"].append({
                "name": name,
                "bases": bases,
                "attributes": attributes,
            })

        # ---- Functions ----
        if node.type in func_types:
            is_async = is_async_function(node)
            is_method_flag = is_method(node, ancestors, class_types)
            name = get_node_name(node, extension) or "<anonymous>"
            params = extract_parameters(node, extension)
            return_type = extract_return_type(node, extension)
            visibility = extract_visibility(node, extension)

            if is_method_flag:
                results["methods"] += 1
            else:
                results["functions"] += 1

            if is_async:
                results["async_functions"] += 1

            results["function_details"].append({
                "name": name,
                "params": params,
                "return_type": return_type,
                "is_async": is_async,
                "visibility": visibility,
                "is_method": is_method_flag,
                # We'll link to class later in the pipeline using parent lookup
            })

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

        for child in node.children:
            traverse(child, ancestors + [node])

    traverse(root_node)
    return results

# ----------------------------------------------------------------------
# Reshape — now returns the detailed lists
# ----------------------------------------------------------------------
def reshape_for_db(parser_metrics):
    return {
        "classes": parser_metrics["classes"],
        "functions": parser_metrics["functions"],
        "methods": parser_metrics["methods"],
        "async_functions": parser_metrics["async_functions"],
        "imports": parser_metrics["import_count"],
        "lines": parser_metrics["lines"],
        "class_details": parser_metrics["class_details"],       # NEW
        "function_details": parser_metrics["function_details"], # NEW
    }

# ----------------------------------------------------------------------
# Parser / tree getters (unchanged)
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
# Demo (unchanged)
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