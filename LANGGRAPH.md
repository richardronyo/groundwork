# Groundwork on LangChain + LangGraph

Every LLM call now goes through a LangChain chat model built by `init_chat_model`. Every multi-step workflow is a LangGraph `StateGraph`. Each existing module still runs from its own CLI, and the pipeline graph composes them.

## Layout

```
kb/llm/client.py                   init_chat_model factory, encrypted key store, structured output, run config
kb/llm/schemas.py                  Pydantic schemas for extraction, critique, key points, tests, reflections
kb/pipeline/ingest_graph.py        NEW  scan → parse → deps → extract → synthesize → embed → translate
kb/vector/extract_business_rules.py    file graph (extract → interrogate → refine) + repo graph (Send fan-out)
kb/agent/test_graph.py             NEW  single-pass and agentic test-generation graphs (shared nodes)
kb/agent/generate_test.py          CLI; generate_test_simple() keeps its old contract
kb/translate/components.py         NEW  RuleInventory, BSG, language-neutral TestSuite, Translation
kb/translate/prompts.py            NEW  analyzer / specifier / test / translator prompts
kb/translate/graph.py              NEW  AgentModernize workflow, reading the KB
kb/verify/pytest_runner.py         NEW  runs generated pytest files (agentic test loop)
kb/verify/harness.py               NEW  runs the same JSON test cases in Python, Node and R
tests/                             NEW  scripted-model tests for every graph (no API key needed)
```

## The graphs

**Pipeline** (`python -m kb.pipeline.ingest_graph <repo>`): only the stages you select are added, in canonical order. A failed stage routes straight to END. Kùzu's single writer is shared by scan and deps, then released so later stages can read the graph.

**Rule extraction**: `load_files ─Send─► extract_file ×N ─► collect ─► synthesize`. Each `extract_file` runs the file graph `extract ─► interrogate ⇄ refine`. With `--refine-rounds 0` (the default) the file graph is a single extraction call, so it behaves exactly as before. With N > 0 it runs the interrogation loop for Experiment 1. `run_extraction()` returns per-file history (every rule set and critique) so you can score each round.

**Test generation**: both modes use the same nodes.
- `single`: `load_context → retrieve → generate → remember`
- `agentic`: adds `run_tests → reflect → generate` until the tests pass or attempts run out.

The ablation is therefore one edge set versus another. A `code` fault from `reflect` means the test found a real bug, so the next attempt marks the test `xfail(strict=True)` rather than weakening it.

**Translation** (`python -m kb.translate.graph <repo> <file> --to javascript`):

```
load → analyze → specify ─Send(node)─► node_tests ×N → collect_tests → translate ⇄ verify → finalize
```

What it reuses from the KB:
- **analyze** is seeded with the file's stored business rules, which it verifies and extends rather than re-deriving.
- **specify** sees the parsed functions/classes and the Kùzu `DEPENDS_ON` files, including their stored BSGs. It emits `depends_on` edges to `<file>::<node>`, and saves the BSG to `bsg_specs`.

Translating dependencies first therefore gives later files cross-file BSGs. Test generation also picks up a file's stored BSG automatically (`--no-bsg` turns that off).

The translation loop follows your existing pipeline's rules:
- Expected results come from running the original code, never from the model.
- Each BSG node gets its own test request, retried up to 3 times when the harness can't run a case.
- It stops at 100%, at max rounds (default 5), or when the same failing tests appear twice in a row.
- It keeps the best round.
- Outputs: the translated code, `rules.json`, `bsg.json`, `tests/{unit,behaviour,skipped}_tests.md`, `tests/tests.json`, per-round JSON, `report.md`, and a row in the `translations` table.

Verified languages are python, javascript and r. Any other target is translated in one unverified round.

## Configuration: models live in config, not state

Graphs are invoked with `run_config(model, provider, rate_limit=..., max_concurrency=...)`. Nodes get their model with `llm_from_config(config, temperature)`. Keeping the model out of graph state lets states checkpoint cleanly (`--checkpoint runs.sqlite`).

- `max_concurrency` replaces `--workers`.
- LangChain's `InMemoryRateLimiter` replaces the `RateLimiter` class.
- Provider retries (`max_retries`) replace `call_llm`'s loop.
- `with_structured_output` replaces `parse_json_response`.

Every LLM-facing schema has no defaults and no free-form dicts, so all of them pass OpenAI strict mode. Test arguments travel as an `args_json` string for this reason.

## Breaking changes and migration

- **LiteLLM is gone.** Install the LangChain integration for each provider you use (`langchain-openai`, `langchain-anthropic`, ...). Stored keys keep working. `"openai/gpt-4o"` and `"openai:gpt-4o"` model strings are still accepted.
- **Memory is now keyed by the repo-prefixed path**, the same key RAG, `business_rules`, Chroma and Kùzu use. Previously `generate_test.py` wrote memory under the unprefixed path, so memory and RAG for one file sat under two different keys. To move old rows:
  ```sql
  UPDATE agent_memory SET file_path = repository_name || '/' || file_path
  WHERE file_path IS NOT NULL AND file_path NOT LIKE repository_name || '/%';
  ```
- **New tables** `bsg_specs` and `translations` are created by `init_db()` and included in `clear_repo` / `clear_all` / `drop_all_tables`.
- **`ENCRYPTION_KEY` is read on first use**, not at import time, so read-only tools no longer exit without it.

## Fixes made along the way

- **Parser parameters**: Python parameters were always empty, because bare parameters are `identifier` nodes and the typed/default variants weren't matched.
- **Parser imports**: `from x import y` recorded `y` instead of `x`, which broke `deps --from-db`.
- **Methods**: methods now carry their enclosing class, and `structure_for_db()` nests them the way `save_structure()` expects.
- **Grammars**: `parser_dicts.py` loads grammars one by one, so a missing package disables only its extensions.
- **Scan without `tree`**: `tree_to_json.files_from_walk()` produces the same rows as the `tree -fi` parser, plus `size_bytes`, without needing `tree`.
- **R support**: R is registered as a language (`.R` files, `source()` dependencies) for the translation pipeline.
- **Memory and RAG degrade gracefully** when chromadb or sentence-transformers aren't installed. The Postgres record is still written.

## Testing

```
pytest tests -q        # needs local Postgres; node for the JS translation tests
```

`tests/fakes.py` has `ScriptedChatModel`, which you can pass as `run_config(llm=...)` to drive any graph through every branch without API calls.
