#!/usr/bin/env bash
# =============================================================================
# Groundwork — Full Ingestion Pipeline
# 1. Asks for repository path
# 2. Runs tree | tree_to_json.py  → repo-tree-<name>.json
# 3. Runs ingest_graph.py         → populates File/Directory nodes
# 4. Runs build_dependencies.py   → populates DEPENDS_ON edges
# 5. Runs extract_rules.py        → business_rules.json + repo_function.json
#
# Usage: ./ingest.sh
#    or: ./ingest.sh /path/to/repo
# =============================================================================

set -euo pipefail

# ── Resolve script directory so we can find sibling scripts ──────────────────

TREE_TO_JSON="kb/graph/tree_to_json.py"
INGEST_GRAPH="kb/graph/json_to_graph.py"
BUILD_DEPS="kb/graph/file_dependencies.py"
EXTRACT_RULES="kb/vector/extract_business_rules.py"

# ── Check dependencies ────────────────────────────────────────────────────────

check_deps() {
  local missing=0

  if ! command -v tree &>/dev/null; then
    echo "  ✗ 'tree' not found. Install with: brew install tree"
    missing=1
  fi

  if ! command -v python3 &>/dev/null; then
    echo "  ✗ 'python3' not found."
    missing=1
  fi

  for script in "$TREE_TO_JSON" "$INGEST_GRAPH" "$BUILD_DEPS" "$EXTRACT_RULES"; do
    if [[ ! -f "$script" ]]; then
      echo "  ✗ Missing script: $script"
      missing=1
    fi
  done

  if (( missing )); then
    echo ""
    echo "  Fix the above and re-run."
    exit 1
  fi
}

# ── Helpers ───────────────────────────────────────────────────────────────────

print_header() {
  echo ""
  printf '%0.s═' {1..60}; echo
  echo "  GROUNDWORK — Ingestion Pipeline"
  printf '%0.s═' {1..60}; echo
  echo ""
}

print_step() {
  local step="$1"
  local label="$2"
  echo ""
  printf '%0.s─' {1..60}; echo
  echo "  Step $step — $label"
  printf '%0.s─' {1..60}; echo
  echo ""
}

# ── Get repo path ─────────────────────────────────────────────────────────────

print_header
check_deps

if [[ $# -ge 1 ]]; then
  REPO_PATH="${1%/}"
else
  printf "  Enter repository path: "
  read -r REPO_PATH
  REPO_PATH="${REPO_PATH%/}"
fi

# Resolve to absolute path
REPO_PATH="$(cd "$REPO_PATH" 2>/dev/null && pwd)" || {
  echo "  ✗ Directory not found: $REPO_PATH"
  exit 1
}

REPO_NAME="$(basename "$REPO_PATH")"
OUTPUT_JSON="repo-tree-${REPO_NAME}.json"

echo ""
echo "  Repository : $REPO_PATH"
echo "  Repo name  : $REPO_NAME"
echo "  Output JSON: $OUTPUT_JSON"

# ── Confirm ───────────────────────────────────────────────────────────────────

echo ""
printf "  Proceed? [Y/n]: "
read -r confirm
confirm="${confirm:-Y}"
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "  Aborted."
  exit 0
fi

# ── Step 1: tree → JSON ───────────────────────────────────────────────────────

print_step 1 "Scanning repository → $OUTPUT_JSON"

tree -f "$REPO_PATH" | python3 "$TREE_TO_JSON" > "$OUTPUT_JSON"

FILE_COUNT=$(python3 -c "import json; d=json.load(open('$OUTPUT_JSON')); print(len(d))")
echo "  ✓ $FILE_COUNT files written to $OUTPUT_JSON"

# ── Step 2: Populate File/Directory nodes ─────────────────────────────────────

print_step 2 "Ingesting file structure into Neo4j"

python3 "$INGEST_GRAPH" "$OUTPUT_JSON" --clear

# ── Step 3: Build DEPENDS_ON edges ───────────────────────────────────────────

print_step 3 "Building dependency edges"

python3 "$BUILD_DEPS" "$OUTPUT_JSON" --repo "$REPO_PATH"

# ── Step 4: Extract business rules ───────────────────────────────────────────

print_step 4 "Extracting business rules and repository function"

python3 "$EXTRACT_RULES" "$OUTPUT_JSON" --repo "$REPO_PATH"

# ── Done ──────────────────────────────────────────────────────────────────────

print_done() {
  echo ""
  printf '%0.s═' {1..60}; echo
  echo "  ✓ Pipeline complete"
  echo ""
  echo "  JSON tree      : $OUTPUT_JSON"
  echo "  Business rules : business_rules.json"
  echo "  Repo function  : repo_function.json"
  echo "  Neo4j          : https://console.neo4j.io"
  echo ""
  echo "  Useful Cypher queries:"
  echo "    MATCH (n) RETURN n"
  echo "    MATCH (a:File)-[:DEPENDS_ON]->(b:File) RETURN a.name, b.name"
  echo "    MATCH (a:File)-[:DEPENDS_ON]->(b:File) RETURN a, b"
  printf '%0.s═' {1..60}; echo
  echo ""
}

print_done