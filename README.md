# Groundwork

AI-based knowledge extraction, testing, and artifact generation for
software repositories.

Groundwork ingests a codebase into a multi-store knowledge base — file
metadata and structure, a file-level dependency graph, LLM-extracted
business rules, and semantic vectors — then uses that knowledge base to
generate unit tests, weakness reports, and diagrams grounded in the
repository's actual code rather than guesses.

## What it does

Writing comprehensive test suites gets harder as a codebase grows:
developers need context on linked files, what a repository actually does,
and where its weak points are. Groundwork automates gathering that context
and uses it to generate:

- **Unit tests** — grounded in the target's real dependency signatures and
  an existing test file's style/framework, not invented from scratch
- **Weakness reports** — flagging likely gaps once tests are generated
- **UML class diagrams** and **dependency graphs** — rendered as images or
  Mermaid, from the same indexed structure
- **Whole-repository reports** — module dependencies, cross-module function
  relationships, and capability landscape

## Architecture

Three layers:

1. **Front-End** — a terminal UI (`groundwork_tui.py`, built on Textual)
   covering ingestion, diagrams, reports, test generation, and browsing the
   knowledge base, all in one place.
2. **Orchestration Layer** — Python modules that coordinate reads across
   the knowledge base (and the LLM) to answer a question or produce an
   artifact: `grab_context.py` (retrieval), `generate_tests.py` (test
   generation), `kb/diagram.py` (diagrams), `system_report.py` (reports).
3. **Knowledge Base Layer** — three stores, each with one job:
   - **PostgreSQL** — file metrics, real class/function names and
     signatures, business rules, and synthesized key points
   - **Kùzu** (embedded graph DB) — file/directory structure and
     `DEPENDS_ON` dependency edges
   - **ChromaDB** — per-repository semantic vectors, one dimension per key
     point, used for context retrieval

A repository is ingested via `ingestion_pipeline.sh`, which runs six
resumable stages (`init → scan → deps → rules → synth → embed`) to
populate all three stores.

## Installation

Requires **Python 3.13**, a running local **PostgreSQL** instance, and the
**`tree`** command-line utility, in addition to the Python packages below.

```bash
# 1. Clone the repository
git clone <repository-url>
cd groundwork

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run Groundwork
python3 groundwork_tui.py
```

The TUI covers everything — ingestion, diagrams, reports, test generation,
and browsing the knowledge base. Individual stages can also be run
directly from the command line, e.g.:

```bash
./ingestion_pipeline.sh <path-to-repo-or-github-url>
python3 -m kb.diagram --repo flask --type class --file flask/src/flask/app.py
python3 generate_tests.py --repo flask --file src/flask/app.py
```

## Usage

Once a repository is ingested, query the knowledge base directly:

```bash
python3 grab_context.py "How does session handling work?" --repo flask --ask
```

Or open the TUI (`python3 groundwork_tui.py`) and use the **Browse** tab to
explore what's stored, the **Diagrams** tab to generate a UML class or
dependency diagram, or the **Tests** tab to generate unit tests for a file.