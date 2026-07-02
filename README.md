# Groundwork

Groundwork ingests a software repository, builds a persistent, queryable knowledge base about it, and uses that knowledge base to answer questions and generate unit tests.

It analyzes a codebase across three coordinated stores, with **PostgreSQL as the single source of truth**:

- **PostgreSQL** — file metrics, business rules, and repo-level key points
- **Neo4j** — file/directory structure and dependency (`DEPENDS_ON`) edges
- **ChromaDB** — key-point-aligned vectors (one collection per repository)

Once a repo is ingested, you can ask what the codebase does, retrieve context for a specific feature, and generate tests grounded in both the source code and the extracted business rules.

---

## How it works

Ingestion runs in six stages. Each reads its inputs from PostgreSQL, so any stage can be run on its own and the pipeline can resume from any point.

| Stage   | What it does                                                        | Writes to            |
|---------|--------------------------------------------------------------------|----------------------|
| `init`  | Creates databases and tables; applies schema migrations            | PostgreSQL           |
| `scan`  | Parses file metrics; builds the file/directory graph               | PostgreSQL + Neo4j   |
| `deps`  | Extracts imports and builds `DEPENDS_ON` edges                     | Neo4j                |
| `rules` | Extracts business rules from each file via OpenAI                  | PostgreSQL           |
| `synth` | Synthesizes repo-level key points from all rules                  | PostgreSQL           |
| `embed` | Builds BERTScore vectors aligned to key points                    | ChromaDB             |

Everything is scoped by repository name, so a single knowledge base can hold many repos side by side.

---

## Prerequisites

- Python 3.13 with a virtualenv
- PostgreSQL running locally (database `repo_analysis`)
- A Neo4j instance (e.g. AuraDB free tier)
- The `tree` command line tool
- `git` (only needed to ingest from a GitHub URL)

Environment variables (in a `.env` file at the project root):

```
POSTGRES_PASSWORD=...
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
OPENAI_API_KEY=...
```

Python dependencies:

```bash
pip install psycopg[binary] neo4j chromadb openai bert-score \
            torch transformers python-dotenv \
            langchain langchain-core langchain-openai
```

**Run every command from the project root** — the directory containing `kb/`. Scripts are invoked as modules (`python3 -m kb.x.y`) so the package imports resolve.

---

## Project layout

```
groundwork/
├── kb/
│   ├── __init__.py
│   ├── graph/
│   │   ├── tree_to_json.py
│   │   ├── json_to_graph.py
│   │   └── file_dependencies.py
│   ├── relationaldb/
│   │   ├── initialize_db.py
│   │   ├── parser.py
│   │   └── metadata.py
│   ├── vector/
│   │   ├── extract_business_rules.py
│   │   ├── synthesize.py
│   │   └── embeddings.py
│   ├── ingest.py
│   ├── grab_context.py
│   ├── generate_tests.py
│   └── clear.py
├── ingestion_pipeline.sh
├── knowledge_base_inspector.ipynb
├── repos/                     # cloned GitHub repos land here
├── chroma_db/                 # ChromaDB persistence
└── .env
```

---

## 1. Ingesting a repository

### Using the shell pipeline

The pipeline accepts either a local path or a GitHub URL. A URL is cloned into `./repos/<name>` first (or pulled if already present).

```bash
# Full ingestion from a local path
./ingestion_pipeline.sh ./flask

# Full ingestion from a GitHub URL
./ingestion_pipeline.sh https://github.com/gothinkster/flask-realworld-example-app

# Prompt for the path/URL interactively
./ingestion_pipeline.sh
```

**Resuming and running individual stages:**

```bash
# Resume from a stage onward (runs it and everything after)
./ingestion_pipeline.sh ./flask --from rules

# Run only one stage
./ingestion_pipeline.sh ./flask --only embed

# Resume rule extraction, skipping files already processed
./ingestion_pipeline.sh ./flask --from rules --resume-rules
```

Valid stage names for `--from` and `--only`: `init`, `scan`, `deps`, `rules`, `synth`, `embed`.

### Using the Python entry point

`kb.ingest` does the same work (cloning + stages) as a scriptable, non-interactive command.

```bash
# From a GitHub URL or a local path
python3 -m kb.ingest https://github.com/gothinkster/flask-realworld-example-app
python3 -m kb.ingest ./flask

# Resume / single stage / skip processed files
python3 -m kb.ingest ./flask --from rules
python3 -m kb.ingest ./flask --only embed
python3 -m kb.ingest ./flask --resume-rules

# Limit lines read per file during rule extraction
python3 -m kb.ingest ./flask --lines 120
```

### Running individual stage modules directly

Each stage can also be run on its own:

```bash
python3 -m kb.relationaldb.initialize_db
python3 -m kb.relationaldb.metadata ./flask
python3 -m kb.graph.json_to_graph <tree.json> --repo flask --clear
python3 -m kb.graph.file_dependencies <tree.json> --repo ./flask
python3 -m kb.vector.extract_business_rules --repo flask --repo-path ./flask --only-unprocessed
python3 -m kb.vector.synthesize --repo flask
python3 -m kb.vector.embeddings --repo flask
```

---

## 2. Inspecting the knowledge base

### List repositories

```bash
# Via the query tool
python3 -m grab_context --list-repos

# Via the clear tool (also lists, then exits)
python3 -m kb.clear --list
```

### Query the knowledge base

`kb.grab_context` retrieves context for a question and can ask an LLM to answer. It auto-detects whether a question is **repo-level** (about the whole codebase) or **file-level** (about a specific feature).

```bash
# Repo-level: infers what the codebase is (auto-runs the inference)
python3 -m kb.grab_context "What is this codebase simulating?" --repo flask

# File-level: retrieves the most relevant files + their rules
python3 -m kb.grab_context "How does authentication work?" --repo flask

# File-level with a full LLM answer
python3 -m kb.grab_context "How does session handling work?" --repo flask --ask
```

**Options:**

| Option              | Meaning                                                            |
|---------------------|--------------------------------------------------------------------|
| `--repo <name>`     | Which repository to query (required if more than one exists)        |
| `--mode <mode>`     | `auto` (default), `repo` (force repo-level), or `file` (force file) |
| `--top <n>`         | Number of files to retrieve for file-level questions                |
| `--min-score <f>`   | Minimum BERTScore threshold for a match (default 0.3)               |
| `--ask`             | Also generate an LLM answer for file-level questions                |
| `--list-repos`      | List repositories in the knowledge base and exit                    |

Force a mode when auto-detection guesses wrong:

```bash
python3 -m kb.grab_context "overview of the payment flow" --repo flask --mode repo
python3 -m kb.grab_context "how does X work?" --repo flask --mode file
```

### Inspect all three stores in a notebook

`knowledge_base_inspector.ipynb` views everything — PostgreSQL tables, Neo4j nodes/edges (with an optional dependency graph plot), ChromaDB vectors, and a cross-store join for a single file.

```bash
jupyter notebook knowledge_base_inspector.ipynb
```

---

## 3. Generating unit tests

`kb.generate_tests` writes tests for a file or a single function, grounded in both the **source code** and the **business rules** from the knowledge base. The test framework is inferred from the file's language (pytest, Jest, JUnit, xUnit, and so on).

```bash
# Tests for a whole file
python3 -m kb.generate_tests --repo flask --file src/app.py

# Tests for a single function
python3 -m kb.generate_tests --repo flask --file src/auth.py --function login

# Preview in the terminal instead of writing a file
python3 -m kb.generate_tests --repo flask --file src/app.py --print

# Write to a specific directory
python3 -m kb.generate_tests --repo flask --file src/app.py --out tests/
```

**Options:**

| Option               | Meaning                                                         |
|----------------------|-----------------------------------------------------------------|
| `--repo <name>`      | Repository name (required if more than one exists)              |
| `--file <path>`      | Target file, relative to the repo root (required)              |
| `--function <name>`  | Target a single function/method within the file                 |
| `--repo-path <path>` | Repo location on disk (default: `./repos/<repo>`)              |
| `--out <dir>`        | Output directory (default: `./generated_tests`)                |
| `--print`            | Print to stdout instead of writing a file                       |

Generated tests are a starting point, not ground truth — review the assertions before trusting them, especially those derived from business rules.

---

## 4. Clearing the knowledge base

`kb.clear` removes data from all three stores. It asks for confirmation unless `--yes` is given.

```bash
# Wipe everything (all repos, all stores)
python3 -m kb.clear

# Remove a single repository from all three stores
python3 -m kb.clear --repo flask

# Skip the confirmation prompt
python3 -m kb.clear --yes

# List what's in the knowledge base, then exit
python3 -m kb.clear --list

# Limit to one store
python3 -m kb.clear --postgres-only
python3 -m kb.clear --neo4j-only
python3 -m kb.clear --chroma-only
```

---

## Command quick reference

```bash
# ── Ingest ────────────────────────────────────────────────
./ingestion_pipeline.sh <path-or-github-url>
./ingestion_pipeline.sh <repo> --from rules
./ingestion_pipeline.sh <repo> --only embed
./ingestion_pipeline.sh <repo> --from rules --resume-rules
python3 -m kb.ingest <path-or-github-url>
python3 -m kb.ingest <repo> --from rules --resume-rules

# ── Inspect / query ───────────────────────────────────────
python3 -m kb.grab_context --list-repos
python3 -m kb.grab_context "What is this codebase?" --repo <name>
python3 -m kb.grab_context "How does X work?" --repo <name> --ask
python3 -m kb.grab_context "..." --repo <name> --mode file --top 3
jupyter notebook knowledge_base_inspector.ipynb

# ── Generate tests ────────────────────────────────────────
python3 -m kb.generate_tests --repo <name> --file <path>
python3 -m kb.generate_tests --repo <name> --file <path> --function <fn>
python3 -m kb.generate_tests --repo <name> --file <path> --print
python3 -m kb.generate_tests --repo <name> --file <path> --out tests/

# ── Clear ─────────────────────────────────────────────────
python3 -m kb.clear --list
python3 -m kb.clear --repo <name>
python3 -m kb.clear                     # everything, with confirmation
python3 -m kb.clear --yes               # everything, no prompt
```

---

## Typical workflow

```bash
# 1. Ingest a repo from GitHub
./ingestion_pipeline.sh https://github.com/gothinkster/flask-realworld-example-app

# 2. Confirm it's in the knowledge base
python3 -m kb.grab_context --list-repos

# 3. Ask what it is
python3 -m kb.grab_context "What is this codebase?" --repo flask-realworld-example-app

# 4. Dig into a feature
python3 -m kb.grab_context "How are articles created?" --repo flask-realworld-example-app --ask

# 5. Generate tests for a file
python3 -m kb.generate_tests --repo flask-realworld-example-app --file conduit/articles/models.py
```