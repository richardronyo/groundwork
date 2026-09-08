import tree_sitter_agda as TSAGDA
import tree_sitter_bash as TSBASH
import tree_sitter_c as TSC
import tree_sitter_cpp as TSCPP
import tree_sitter_c_sharp as TSC_SHARP
import tree_sitter_css as TSCSS
import tree_sitter_embedded_template as TSEMBEDDED_TEMPLATE
import tree_sitter_go as TSGO
import tree_sitter_haskell as TSHASKELL
import tree_sitter_html as TSHTML
import tree_sitter_java as TSJAVA
import tree_sitter_javascript as TSJAVASCRIPT
import tree_sitter_json as TSJSON
import tree_sitter_julia as TSJULIA
import tree_sitter_kotlin as TSKOTLIN
import tree_sitter_lua as TSLUA
import tree_sitter_ocaml as TSOCAML
import tree_sitter_php as TSPHP
import tree_sitter_python as TSPYTHON
import tree_sitter_ruby as TSRUBY
import tree_sitter_rust as TSRUST
import tree_sitter_scala as TSSCALA
import tree_sitter_typescript as TSTYPESCRIPT
import tree_sitter_verilog as TSVERILOG

EXTENSION_TO_MODULE = {
    ".agda": TSAGDA,
    ".sh": TSBASH,
    ".bash": TSBASH,
    ".c": TSC,
    ".h": TSC,
    ".cpp": TSCPP,
    ".cs": TSC_SHARP,
    ".css": TSCSS,
    ".erb": TSEMBEDDED_TEMPLATE,
    ".go": TSGO,
    ".hs": TSHASKELL,
    ".html": TSHTML,
    ".java": TSJAVA,
    ".js": TSJAVASCRIPT,
    ".json": TSJSON,
    ".jl": TSJULIA,
    ".kt": TSKOTLIN,
    ".lua": TSLUA,
    ".ml": TSOCAML,
    ".mli": TSOCAML,
    ".php": TSPHP,
    ".py": TSPYTHON,
    ".rb": TSRUBY,
    ".rs": TSRUST,
    ".scala": TSSCALA,
    ".ts": TSTYPESCRIPT,
    ".tsx": TSTYPESCRIPT,
    ".v": TSVERILOG
}

FUNCTION_TYPES = {
    ".agda": ["function_definition"],
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