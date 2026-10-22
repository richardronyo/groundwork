"""
Tree-sitter grammar registry. Grammars are imported one by one so a missing
package disables only its own extensions (with a warning on first use)
instead of making the whole parser unimportable.
"""

import importlib

_GRAMMAR_PACKAGES = {
    ".agda": "tree_sitter_agda",
    ".jsx": "tree_sitter_javascript",  # tree-sitter-javascript parses JSX fine; no separate grammar needed
    ".sh": "tree_sitter_bash",
    ".bash": "tree_sitter_bash",
    ".c": "tree_sitter_c",
    ".h": "tree_sitter_c",
    ".cpp": "tree_sitter_cpp",
    ".cs": "tree_sitter_c_sharp",
    ".css": "tree_sitter_css",
    ".erb": "tree_sitter_embedded_template",
    ".go": "tree_sitter_go",
    ".hs": "tree_sitter_haskell",
    ".html": "tree_sitter_html",
    ".java": "tree_sitter_java",
    ".js": "tree_sitter_javascript",
    ".json": "tree_sitter_json",
    ".jl": "tree_sitter_julia",
    ".kt": "tree_sitter_kotlin",
    ".lua": "tree_sitter_lua",
    ".ml": "tree_sitter_ocaml",
    ".mli": "tree_sitter_ocaml",
    ".php": "tree_sitter_php",
    ".py": "tree_sitter_python",
    ".rb": "tree_sitter_ruby",
    ".rs": "tree_sitter_rust",
    ".scala": "tree_sitter_scala",
    ".ts": "tree_sitter_typescript",
    ".tsx": "tree_sitter_typescript",
    ".v": "tree_sitter_verilog",
}

EXTENSION_TO_MODULE = {}
MISSING_GRAMMARS = {}          # extension -> package name that failed to import
_loaded = {}
for _ext, _pkg in _GRAMMAR_PACKAGES.items():
    if _pkg not in _loaded:
        try:
            _loaded[_pkg] = importlib.import_module(_pkg)
        except ImportError:
            _loaded[_pkg] = None
    if _loaded[_pkg] is not None:
        EXTENSION_TO_MODULE[_ext] = _loaded[_pkg]
    else:
        MISSING_GRAMMARS[_ext] = _pkg

FUNCTION_TYPES = {
    ".agda": ["function_definition"],
    ".jsx": ["function_declaration", "function_expression", "arrow_function"],
    ".sh": ["function_definition"],
    ".bash": ["function_definition"],
    ".c": ["function_definition"],
    ".h": ["function_definition"],
    ".cpp": ["function_definition"],
    ".cs": ["method_declaration"],
    ".css": [],
    ".erb": [],
    ".go": ["function_declaration"],
    ".hs": ["function"],
    ".html": [],
    ".java": ["method_declaration"],
    ".js": ["function_declaration", "function_expression", "arrow_function"],
    ".json": [],
    ".jl": ["function_definition"],
    ".kt": ["function_declaration"],
    ".lua": ["function_declaration"],
    ".ml": ["let_binding"],
    ".mli": ["let_binding"],
    ".php": ["function_definition"],
    ".py": ["function_definition"],
    ".rb": ["method"],
    ".rs": ["function_item"],
    ".scala": ["method"],
    ".ts": ["function_declaration", "function_expression", "arrow_function"],
    ".tsx": ["function_declaration", "function_expression", "arrow_function"],
    ".v": ["function_declaration"]
}

CLASS_TYPES = {
    ".agda": ["record"],
    ".jsx": ["class_declaration"],
    ".sh": [],
    ".bash": [],
    ".c": [],
    ".h": [],
    ".cpp": ["class_specifier"],
    ".cs": ["class_declaration"],
    ".css": [],
    ".erb": [],
    ".go": ["type_declaration"],
    ".hs": ["class_declaration"],
    ".html": [],
    ".java": ["class_declaration"],
    ".js": ["class_declaration"],
    ".json": [],
    ".jl": ["struct_definition"],
    ".kt": ["class_declaration"],
    ".lua": [],
    ".ml": ["class_declaration"],
    ".mli": ["class_declaration"],
    ".php": ["class_declaration"],
    ".py": ["class_definition"],
    ".rb": ["class"],
    ".rs": ["struct_item", "enum_item"],
    ".scala": ["class_definition"],
    ".ts": ["class_declaration"],
    ".tsx": ["class_declaration"],
    ".v": ["class_declaration"]
}

IMPORT_TYPES = {
    ".agda": ["import"],
    ".jsx": ["import_statement"],
    ".sh": [],
    ".bash": [],
    ".c": ["preproc_include"],
    ".h": ["preproc_include"],
    ".cpp": ["preproc_include"],
    ".cs": ["using_directive"],
    ".css": ["import"],
    ".erb": [],
    ".go": ["import_declaration"],
    ".hs": ["import"],
    ".html": [],
    ".java": ["import_declaration"],
    ".js": ["import_statement"],
    ".json": [],
    ".jl": ["import"],
    ".kt": ["import_header"],
    ".lua": [],
    ".ml": ["open", "include"],
    ".mli": ["open", "include"],
    ".php": ["namespace_use"],
    ".py": ["import_statement", "import_from_statement"],
    ".rb": [],
    ".rs": ["use_declaration"],
    ".scala": ["import_declaration"],
    ".ts": ["import_statement"],
    ".tsx": ["import_statement"],
    ".v": ["import"]
}

COMMENT_TYPES = {
    ".agda": ["comment"],
    ".jsx": ["comment"],
    ".sh": ["comment"],
    ".bash": ["comment"],
    ".c": ["comment"],
    ".h": ["comment"],
    ".cpp": ["comment"],
    ".cs": ["comment"],
    ".css": ["comment"],
    ".erb": ["comment"],
    ".go": ["comment"],
    ".hs": ["comment"],
    ".html": ["comment"],
    ".java": ["comment"],
    ".js": ["comment"],
    ".json": [],
    ".jl": ["comment"],
    ".kt": ["comment"],
    ".lua": ["comment"],
    ".ml": ["comment"],
    ".mli": ["comment"],
    ".php": ["comment"],
    ".py": ["comment"],
    ".rb": ["comment"],
    ".rs": ["line_comment", "block_comment"],
    ".scala": ["comment"],
    ".ts": ["comment"],
    ".tsx": ["comment"],
    ".v": ["comment"]
}

INTERFACE_TYPES = {
    ".agda": [],
    ".jsx": [],
    ".sh": [],
    ".bash": [],
    ".c": [],
    ".h": [],
    ".cpp": [],
    ".cs": ["interface_declaration"],
    ".css": [],
    ".erb": [],
    ".go": ["interface_type"],
    ".hs": [],
    ".html": [],
    ".java": ["interface_declaration"],
    ".js": [],
    ".json": [],
    ".jl": [],
    ".kt": ["interface_declaration"],
    ".lua": [],
    ".ml": [],
    ".mli": [],
    ".php": ["interface_declaration"],
    ".py": [],
    ".rb": [],
    ".rs": ["trait_item"],
    ".scala": ["trait_def"],
    ".ts": ["interface_declaration"],
    ".tsx": ["interface_declaration"],
    ".v": ["interface"]
}