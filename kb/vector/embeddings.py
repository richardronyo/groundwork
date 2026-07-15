"""
Groundwork — Stage 5: ChromaDB Vector Store (PostgreSQL-backed)

Reads business rules and key points from PostgreSQL. For each file builds a
(n_keypoints,) vector where dimension k = average BERTScore F1 of the file's
rules against key point k. Stores vectors + metadata in ChromaDB.

Usage:
    python3 embeddings.py --repo flask
    python3 embeddings.py            # if only one repo in the DB
    python3 embeddings.py --repo flask --workers 8

Requirements:
    pip install chromadb psycopg python-dotenv bert-score torch transformers
"""

import json
import sys
import argparse
import hashlib
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import chromadb
from dotenv import load_dotenv

from kb.relationaldb.initialize_db import (
    get_connection, load_business_rules_from_db,
    load_key_points_from_db, list_repositories,
)

load_dotenv()

CHROMA_COLLECTION = "groundwork"  # base name; per-repo becomes groundwork_<repo>
CHROMA_DB_PATH    = "./chroma_db"
BERT_MODEL        = "distilbert-base-uncased"
CACHE_DIR         = Path("./cache/vectors")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def collection_name_for(repo_name: str) -> str:
    """Per-repo collection name. Each repo has its own collection because the
    vector dimension (= key-point count) differs per repo, and ChromaDB locks
    dimension per collection."""
    safe = "".join(c if c.isalnum() else "_" for c in repo_name.lower())
    return f"{CHROMA_COLLECTION}_{safe}"


# ── Fast path: encode once, compare by matrix multiply ────────────────────────
#
# The BERTScore path calls the model once per (file × key point) pair, so every
# rule is re-encoded once per key point — O(files × key_points) forward passes.
# Instead we encode each unique rule ONCE and each key point ONCE, then compute
# all similarities with a single matrix multiply: O(rules + key_points) passes.
# The stored vector keeps the same meaning: dimension k = how similar this
# file's rules are, on average, to key point k.

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _encode_sentence_transformers(texts, model_name, batch_size):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size,
                        convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=True)


def _encode_transformers(texts, model_name, batch_size):
    """Fallback: mean-pooled transformer embeddings, L2-normalised."""
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state          # (B, T, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            emb = summed / counts                            # mean pooling
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())
            done = min(i + batch_size, len(texts))
            print(f"\r    encoding {done}/{len(texts)}", end="", flush=True)
    print()
    return np.vstack(out)


def encode_texts(texts, model_name=EMBED_MODEL, batch_size=64):
    """Encodes texts to L2-normalised vectors. Prefers sentence-transformers."""
    try:
        return _encode_sentence_transformers(texts, model_name, batch_size)
    except ImportError:
        print("  (sentence-transformers not installed — using transformers fallback)")
        fallback = BERT_MODEL if "sentence-transformers" in model_name else model_name
        return _encode_transformers(texts, fallback, batch_size)


def build_vectorstore_fast(business_rules, key_points, collection,
                           model_name=EMBED_MODEL, batch_size=64):
    """
    Encode-once vectoriser. Each file's vector[k] = mean cosine similarity of
    that file's rules to key point k — same shape and meaning as the BERTScore
    path, computed in a fraction of the time.
    """
    import numpy as np

    total = len(business_rules)
    n_kp = len(key_points)

    # Flatten rules, remembering which file each belongs to
    all_rules, owners = [], []
    for rel, rules in business_rules.items():
        for r in rules:
            all_rules.append(r)
            owners.append(rel)

    if not all_rules:
        print("  No rules to embed.")
        return

    # De-duplicate: identical rule text only needs encoding once
    uniq = list(dict.fromkeys(all_rules))
    index_of = {t: i for i, t in enumerate(uniq)}

    print(f"\n  Files to process  : {total}")
    print(f"  Key points        : {n_kp}")
    print(f"  Vector dimensions : {n_kp} (one per key point)")
    print(f"  Rules             : {len(all_rules)} ({len(uniq)} unique)")
    print(f"  Model             : {model_name}")
    print(f"\n  Encoding {len(uniq)} unique rules once...")
    R = encode_texts(uniq, model_name, batch_size)          # (U, D) normalised

    print(f"  Encoding {n_kp} key points once...")
    K = encode_texts(list(key_points), model_name, batch_size)  # (K, D) normalised

    # All rule↔key-point cosine similarities in one matrix multiply
    print("  Computing similarities (single matrix multiply)...")
    S = R @ K.T                                             # (U, K)

    # Per-file mean over its rules
    upserted = skipped = 0
    ids, embs, metas, docs = [], [], [], []
    for i, (rel, rules) in enumerate(business_rules.items()):
        pct = int((i + 1) / total * 40)
        bar = "█" * pct + "░" * (40 - pct)
        name = rel.split("/")[-1]
        print(f"\r  [{bar}] {i+1}/{total}  {name:<40}", end="", flush=True)

        if not rules:
            skipped += 1
            continue

        rows = [index_of[r] for r in rules]
        file_vector = [round(float(v), 6) for v in S[rows].mean(axis=0)]

        top_index = int(max(range(n_kp), key=lambda k: file_vector[k]))
        metas.append({
            "relative": rel, "name": name, "rule_count": len(rules),
            "top_kp": key_points[top_index], "top_kp_index": top_index,
            "top_score": round(file_vector[top_index], 4),
            "kp_scores": json.dumps({f"kp_{k}": file_vector[k] for k in range(n_kp)}),
        })
        ids.append(rel)
        embs.append(file_vector)
        docs.append("\n\n".join(rules))
        upserted += 1

        # Batch the upserts so ChromaDB isn't hit once per file
        if len(ids) >= 500:
            collection.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
            ids, embs, metas, docs = [], [], [], []

    if ids:
        collection.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=docs)

    print(f"\n\n  ✓ {upserted} files added to ChromaDB, {skipped} skipped (no rules).")


def load_bert_scorer():
    try:
        from bert_score import score as bert_score_fn
        return bert_score_fn
    except ImportError:
        print("Error: bert-score not installed. Run: pip install bert-score")
        sys.exit(1)


def get_cache_key(rules, key_points):
    """Generate cache key based on rules and key points."""
    content = json.dumps({"rules": sorted(rules), "key_points": key_points}, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def compute_file_vector(rules, key_points, bert_score_fn, use_cache=True):
    """Compute file vector with caching to avoid recomputation."""
    if not rules:
        return []
    
    cache_key = get_cache_key(rules, key_points)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    
    # Compute vector
    file_vector = []
    for kp in key_points:
        refs = [kp] * len(rules)
        _, _, F1 = bert_score_fn(cands=rules, refs=refs,
                                 model_type=BERT_MODEL, verbose=False)
        file_vector.append(round(float(F1.mean()), 6))
    
    # Cache result
    if use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(file_vector, f)
        except Exception:
            pass
    
    return file_vector


def compute_single_file_vector(args):
    """Compute vector for a single file. Must be picklable for ProcessPoolExecutor."""
    file_path, rules, key_points, bert_model, use_cache = args
    
    if not rules:
        return file_path, None, 0, []
    
    try:
        from bert_score import score as bert_score_fn
        
        file_vector = []
        for kp in key_points:
            refs = [kp] * len(rules)
            _, _, F1 = bert_score_fn(cands=rules, refs=refs,
                                     model_type=bert_model, verbose=False)
            file_vector.append(round(float(F1.mean()), 6))
        
        # Calculate top key point
        n_kp = len(key_points)
        top_index = int(max(range(n_kp), key=lambda k: file_vector[k]))
        top_score = round(file_vector[top_index], 4)
        
        # Create metadata
        metadata = {
            "relative": file_path,
            "name": file_path.split("/")[-1],
            "rule_count": len(rules),
            "top_kp": key_points[top_index],
            "top_kp_index": top_index,
            "top_score": top_score,
            "kp_scores": json.dumps({f"kp_{k}": file_vector[k] for k in range(n_kp)}),
        }
        
        # Cache result
        if use_cache:
            cache_key = get_cache_key(rules, key_points)
            cache_file = CACHE_DIR / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(file_vector, f)
            except Exception:
                pass
        
        return file_path, file_vector, metadata, rules
    except Exception as e:
        print(f"\n  Error computing vector for {file_path}: {e}")
        return file_path, None, None, None


def get_collection(db_path, collection_name, expected_dim=None) -> chromadb.Collection:
    """
    Returns the collection, recreating it if its stored vector dimension
    no longer matches expected_dim. ChromaDB locks a collection to the
    dimension of its first insert, so a changed key-point count (which
    changes the vector length) requires a fresh collection.
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"})

    if expected_dim is not None and collection.count() > 0:
        # Peek at one stored vector to read its dimension
        existing = collection.peek(1)
        embeddings = existing.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            stored_dim = len(embeddings[0])
            if stored_dim != expected_dim:
                print(f"  ⚠ Dimension changed ({stored_dim} → {expected_dim}); "
                      f"recreating collection '{collection_name}'.")
                client.delete_collection(collection_name)
                collection = client.get_or_create_collection(
                    name=collection_name, metadata={"hnsw:space": "cosine"})

    return collection


def build_vectorstore_parallel(business_rules, key_points, collection, bert_score_fn, 
                               max_workers=4, use_cache=True):
    """Parallel version of build_vectorstore."""
    total = len(business_rules)
    n_kp = len(key_points)
    print(f"\n  Files to process  : {total}")
    print(f"  Key points        : {n_kp}")
    print(f"  Vector dimensions : {n_kp} (one per key point)")
    print(f"  Using {max_workers} workers for BERTScore computation\n")
    
    # Prepare arguments for parallel processing
    args_list = [
        (file_path, rules, key_points, BERT_MODEL, use_cache)
        for file_path, rules in business_rules.items()
    ]
    
    upserted = skipped = 0
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(compute_single_file_vector, args): args[0] 
            for args in args_list
        }
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            completed += 1
            
            # Update progress
            pct = int(completed / total * 40)
            bar = "█" * pct + "░" * (40 - pct)
            name = file_path.split("/")[-1]
            print(f"\r  [{bar}] {completed}/{total}  {name:<40}", end="", flush=True)
            
            try:
                path, vector, metadata, rules = future.result()
                if vector is None:
                    skipped += 1
                    continue
                
                # Store in ChromaDB
                collection.upsert(
                    ids=[path], 
                    embeddings=[vector],
                    metadatas=[metadata], 
                    documents=["\n\n".join(rules)]
                )
                upserted += 1
            except Exception as e:
                print(f"\n  Error processing {file_path}: {e}")
                skipped += 1
    
    print(f"\n\n  ✓ {upserted} files added to ChromaDB, {skipped} skipped (no rules or errors).")


def build_vectorstore_sequential(business_rules, key_points, collection, bert_score_fn, use_cache=True):
    """Original sequential version for comparison."""
    total = len(business_rules)
    n_kp = len(key_points)
    print(f"\n  Files to process  : {total}")
    print(f"  Key points        : {n_kp}")
    print(f"  Vector dimensions : {n_kp} (one per key point)\n")

    upserted = skipped = 0
    for i, (relative, rules) in enumerate(business_rules.items()):
        pct = int((i + 1) / total * 40)
        bar = "█" * pct + "░" * (40 - pct)
        name = relative.split("/")[-1]
        print(f"\r  [{bar}] {i+1}/{total}  {name:<40}", end="", flush=True)

        if not rules:
            skipped += 1
            continue

        file_vector = compute_file_vector(rules, key_points, bert_score_fn, use_cache)
        top_index = int(max(range(n_kp), key=lambda k: file_vector[k]))
        metadata = {
            "relative": relative, "name": name, "rule_count": len(rules),
            "top_kp": key_points[top_index], "top_kp_index": top_index,
            "top_score": round(file_vector[top_index], 4),
            "kp_scores": json.dumps({f"kp_{k}": file_vector[k] for k in range(n_kp)}),
        }
        collection.upsert(
            ids=[relative], embeddings=[file_vector],
            metadatas=[metadata], documents=["\n\n".join(rules)])
        upserted += 1

    print(f"\n\n  ✓ {upserted} files added to ChromaDB, {skipped} skipped (no rules).")


def print_summary(collection, key_points):
    count = collection.count()
    print(f"\n  ┌─ ChromaDB Summary ───────────────────────────────┐")
    print(f"  │  Collection       : {collection.name:<30}│")
    print(f"  │  File vectors     : {count:<30}│")
    print(f"  │  Vector dimensions: {len(key_points):<30}│")
    print(f"  └──────────────────────────────────────────────────┘\n")


def resolve_repo(conn, requested):
    repos = list_repositories(conn)
    if not repos:
        print("Error: no repositories in DB. Run metadata.py first.")
        sys.exit(1)
    if requested:
        return requested
    if len(repos) == 1:
        print(f"  Using only repository in DB: {repos[0]}")
        return repos[0]
    print("  Multiple repositories — specify one with --repo:")
    for r in repos:
        print(f"    - {r}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Groundwork — build ChromaDB vectors from PostgreSQL")
    parser.add_argument("--repo", help="Repository name as stored in the DB")
    parser.add_argument("--collection", default=None,
                        help="Override collection name (default: groundwork_<repo>)")
    parser.add_argument("--db", default=CHROMA_DB_PATH)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for BERTScore (default: 4)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching of computed vectors")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential processing (bertscore method only)")
    parser.add_argument("--method", choices=["fast", "bertscore"], default="fast",
                        help="fast = encode once + matrix multiply (default); "
                             "bertscore = original per-pair BERTScore (slow)")
    parser.add_argument("--embed-model", default=EMBED_MODEL,
                        help=f"Embedding model for --method fast (default: {EMBED_MODEL})")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size for --method fast (default: 64)")
    args = parser.parse_args()

    conn = get_connection()
    try:
        repo_name = resolve_repo(conn, args.repo)
        print(f"\n  Loading data for '{repo_name}' from PostgreSQL...")
        business_rules = load_business_rules_from_db(conn, repo_name)
        key_points     = load_key_points_from_db(conn, repo_name)

        if not business_rules:
            print(f"Error: no business rules for '{repo_name}'.")
            sys.exit(1)
        if not key_points:
            print(f"Error: no key points for '{repo_name}'. Run synthesize.py first.")
            sys.exit(1)

        print(f"  Loaded {len(business_rules)} files with rules")
        print(f"  Loaded {len(key_points)} key points")
        print(f"  ChromaDB path      : {args.db}")
        print(f"  Method             : {args.method}")
        if args.method == "bertscore":
            print(f"  BERTScore model    : {BERT_MODEL}")
            print(f"  Cache enabled      : {not args.no_cache}")
    finally:
        conn.close()

    coll_name = args.collection or collection_name_for(repo_name)
    collection = get_collection(args.db, coll_name, expected_dim=len(key_points))

    if args.method == "fast":
        build_vectorstore_fast(business_rules, key_points, collection,
                               model_name=args.embed_model, batch_size=args.batch_size)
    else:
        bert_score_fn = load_bert_scorer()
        print(f"  ✓ BERTScore model loaded.\n")
        if args.sequential:
            build_vectorstore_sequential(business_rules, key_points, collection,
                                         bert_score_fn, not args.no_cache)
        else:
            build_vectorstore_parallel(business_rules, key_points, collection,
                                       bert_score_fn, args.workers, not args.no_cache)

    print_summary(collection, key_points)


if __name__ == "__main__":
    main()