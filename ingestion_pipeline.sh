#!/usr/bin/env bash
# =============================================================================
# Groundwork — Unified Ingestion Pipeline (PostgreSQL source of truth)
#
# Run from the PROJECT ROOT (the directory containing kb/).
# Accepts a local path OR a GitHub URL (which it clones into ./repos/).
#
# STAGES (run all, or resume from any one):
#   init  scan  deps  rules  synth  embed
#
# Usage:
#   ./ingestion_pipeline.sh /path/to/repo
#   ./ingestion_pipeline.sh https://github.com/gothinkster/flask-realworld-example-app
#   ./ingestion_pipeline.sh /path/to/repo --from rules
#   ./ingestion_pipeline.sh /path/to/repo --only embed
#   ./ingestion_pipeline.sh /path/to/repo --from rules --resume-rules
#   ./ingestion_pipeline.sh /path/to/repo --workers 10 --embed-workers 8
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

# Module paths (dotted, no .py) — run from project root with python3 -m
TREE_TO_JSON_MOD="kb.graph.tree_to_json"
JSON_TO_GRAPH_MOD="kb.graph.json_to_graph"
FILE_DEPS_MOD="kb.graph.file_dependencies"
INIT_DB_MOD="kb.relationaldb.initialize_db"
METADATA_MOD="kb.relationaldb.metadata"
EXTRACT_RULES_MOD="kb.vector.extract_business_rules"
SYNTHESIZE_MOD="kb.vector.synthesize"
EMBEDDINGS_MOD="kb.vector.embeddings"

STAGES=(init scan deps rules synth embed)

# Default parallelization settings
WORKERS=5
EMBED_WORKERS=4
RATE_LIMIT=60

stage_index() {
    local target="$1"
    for i in "${!STAGES[@]}"; do
        [[ "${STAGES[$i]}" == "$target" ]] && { echo "$i"; return; }
    done
    echo "-1"
}

should_run() {
    local stage="$1"
    local idx
    idx=$(stage_index "$stage")
    [[ "$idx" -ge "$START_IDX" && "$idx" -le "$END_IDX" ]]
}

print_header() { echo ""; echo -e "${BLUE}═══ GROUNDWORK — Ingestion Pipeline ═══${NC}"; echo ""; }
print_step()   { echo ""; echo -e "${YELLOW}━━━ Stage: $1 ━━━${NC}"; echo ""; }
print_success(){ echo -e "${GREEN}  ✓ $1${NC}"; }
print_error()  { echo -e "${RED}  ✗ $1${NC}"; exit 1; }
print_info()   { echo -e "${BLUE}  ℹ $1${NC}"; }

check_dependency() { command -v "$1" &>/dev/null || print_error "$1 not found."; }
check_python_module() {
    python3 -c "import $1" 2>/dev/null || print_error "Python module '$1' missing. pip install $2"
}

# ── Argument parsing ──────────────────────────────────────────────────────────
REPO_PATH=""
FROM_STAGE="init"
ONLY_STAGE=""
RESUME_RULES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)  FROM_STAGE="$2"; shift 2 ;;
        --only)  ONLY_STAGE="$2"; shift 2 ;;
        --resume-rules) RESUME_RULES="--only-unprocessed"; shift ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --embed-workers) EMBED_WORKERS="$2"; shift 2 ;;
        --rate-limit) RATE_LIMIT="$2"; shift 2 ;;
        *)       REPO_PATH="${1%/}"; shift ;;
    esac
done

print_header

# ── Must run from project root (where kb/ lives) ──────────────────────────────
[[ -d "kb" ]] || print_error "Run this from the project root (the directory containing kb/)."

# ── Dependencies ──────────────────────────────────────────────────────────────
print_info "Checking dependencies..."
check_dependency tree
check_dependency python3
check_python_module psycopg "psycopg[binary]"
check_python_module kuzu kuzu
check_python_module chromadb chromadb
check_python_module openai openai
check_python_module bert_score bert-score
check_python_module dotenv python-dotenv
print_success "Dependencies OK"

# ── Repo path (accepts a local path OR a GitHub URL) ──────────────────────────
if [[ -z "$REPO_PATH" ]]; then
    printf "  Enter repository path or GitHub URL: "
    read -r REPO_PATH
    REPO_PATH="${REPO_PATH%/}"
fi

# If it looks like a git URL, clone it into ./repos/<name>
if [[ "$REPO_PATH" =~ ^https?://|^git@|\.git$ ]] || [[ "$REPO_PATH" == *github.com* ]]; then
    check_dependency git
    CLONE_NAME="$(basename "${REPO_PATH%.git}")"
    CLONE_DIR="./repos/$CLONE_NAME"
    mkdir -p ./repos
    if [[ -d "$CLONE_DIR/.git" ]]; then
        print_info "Repo already cloned at $CLONE_DIR — pulling latest..."
        git -C "$CLONE_DIR" pull --ff-only || print_info "Pull skipped (local changes or detached)."
    else
        print_info "Cloning $REPO_PATH ..."
        git clone --depth 1 "$REPO_PATH" "$CLONE_DIR" || print_error "git clone failed"
    fi
    REPO_PATH="$CLONE_DIR"
fi

[[ -d "$REPO_PATH" ]] || print_error "Directory not found: $REPO_PATH"
REPO_PATH="$(cd "$REPO_PATH" && pwd)"
REPO_NAME="$(basename "$REPO_PATH")"

print_info "Repository : $REPO_PATH"
print_info "Repo name  : $REPO_NAME"
print_info "Workers    : $WORKERS (extraction) / $EMBED_WORKERS (embedding)"
print_info "Rate limit : $RATE_LIMIT API calls/minute"

# ── Determine which stages to run ─────────────────────────────────────────────
if [[ -n "$ONLY_STAGE" ]]; then
    START_IDX=$(stage_index "$ONLY_STAGE")
    END_IDX=$START_IDX
    [[ "$START_IDX" == "-1" ]] && print_error "Unknown stage: $ONLY_STAGE"
    print_info "Running ONLY stage: $ONLY_STAGE"
else
    START_IDX=$(stage_index "$FROM_STAGE")
    END_IDX=$((${#STAGES[@]} - 1))
    [[ "$START_IDX" == "-1" ]] && print_error "Unknown stage: $FROM_STAGE"
    print_info "Running stages: ${STAGES[$START_IDX]} → ${STAGES[$END_IDX]}"
fi

echo ""
printf "  Proceed? [Y/n]: "
read -r confirm; confirm="${confirm:-Y}"
[[ "$confirm" =~ ^[Yy]$ ]] || { print_info "Aborted."; exit 0; }

TEMP_JSON=$(mktemp)
trap 'rm -f "$TEMP_JSON"' EXIT

generate_tree() {
    if [[ ! -s "$TEMP_JSON" ]]; then
        print_info "Scanning file tree..."
        tree -f "$REPO_PATH" | python3 -m "$TREE_TO_JSON_MOD" > "$TEMP_JSON"
        local count
        count=$(python3 -c "import json;print(len(json.load(open('$TEMP_JSON'))))")
        print_success "Found $count files"
    fi
}

# ── Stage 1: init ─────────────────────────────────────────────────────────────
if should_run init; then
    print_step "init — initialize databases"
    python3 -m "$INIT_DB_MOD" || print_error "DB init failed"
    mkdir -p ./chroma_db
    print_success "Databases initialized"
fi

# ── Stage 2: scan ─────────────────────────────────────────────────────────────
if should_run scan; then
    print_step "scan — file structure + metrics"
    generate_tree
    print_info "Populating Kùzu file nodes..."
    python3 -m "$JSON_TO_GRAPH_MOD" "$TEMP_JSON" --repo "$REPO_NAME" --clear || print_error "Kùzu node ingest failed"
    print_info "Saving file metrics to PostgreSQL..."
    python3 -m "$METADATA_MOD" "$REPO_PATH" || print_error "Metrics ingest failed"
    print_success "Scan complete"
fi

# ── Stage 3: deps ─────────────────────────────────────────────────────────────
if should_run deps; then
    print_step "deps — dependency edges"
    generate_tree
    python3 -m "$FILE_DEPS_MOD" "$TEMP_JSON" --repo "$REPO_PATH" --repo-name "$REPO_NAME" \
        || print_error "Dependency edges failed"
    print_success "Dependencies built"
fi

# ── Stage 4: rules ────────────────────────────────────────────────────────────
if should_run rules; then
    print_step "rules — business rules (OpenAI → PostgreSQL)"
    python3 -m "$EXTRACT_RULES_MOD" --repo "$REPO_NAME" --repo-path "$REPO_PATH" \
        --no-synthesize $RESUME_RULES --workers "$WORKERS" --rate-limit "$RATE_LIMIT" \
        || print_error "Rule extraction failed"
    print_success "Business rules extracted"
fi

# ── Stage 5: synth ────────────────────────────────────────────────────────────
if should_run synth; then
    print_step "synth — key points (OpenAI → PostgreSQL)"
    python3 -m "$SYNTHESIZE_MOD" --repo "$REPO_NAME" || print_error "Synthesis failed"
    print_success "Key points synthesized"
fi

# ── Stage 6: embed ────────────────────────────────────────────────────────────
if should_run embed; then
    print_step "embed — BERTScore vectors → ChromaDB"
    python3 -m "$EMBEDDINGS_MOD" --repo "$REPO_NAME" --workers "$EMBED_WORKERS" \
        || print_error "Embedding failed"
    print_success "Vector store built"
fi

echo ""
echo -e "${GREEN}═══ ✓ PIPELINE COMPLETE ═══${NC}"
echo ""
echo -e "${BLUE}  PostgreSQL:${NC} repo_analysis (files, business_rules, key_points)"
echo -e "${BLUE}  Kùzu:${NC} ./kuzu_db — File nodes + CONTAINS + SAME_DIR + DEPENDS_ON"
echo -e "${BLUE}  ChromaDB:${NC} groundwork_${REPO_NAME} collection"
echo ""
echo -e "${BLUE}  Query it:${NC}"
echo "    python3 -m kb.grab_context \"What is this codebase?\" --repo $REPO_NAME"
echo ""
echo -e "${BLUE}  Resume examples:${NC}"
echo "    ./ingestion_pipeline.sh $REPO_PATH --from rules"
echo "    ./ingestion_pipeline.sh $REPO_PATH --only embed"
echo ""