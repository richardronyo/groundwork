#!/usr/bin/env python3
"""
Groundwork — Multi-Database Context Retrieval

Given a user prompt, this script:
1. Uses BERTScore to compare the prompt against all key points
2. Finds the most relevant key point
3. Gets the top 2 files that match this key point from ChromaDB (using metadata filter)
4. Retrieves all DEPENDS_ON dependencies from Neo4j
5. Fetches business rules from PostgreSQL

Usage:
    python3 grab_context.py "How does user authentication work?"
    python3 grab_context.py --prompt "payment processing" --top 3
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

import chromadb
import psycopg
from neo4j import GraphDatabase
from dotenv import load_dotenv
import torch

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION = "groundwork"
BERT_MODEL = "distilbert-base-uncased"

# PostgreSQL config
DB_CONFIG = {
    "host": "localhost",
    "dbname": "repo_analysis",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
}

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ── BERTScore ─────────────────────────────────────────────────────────────────

def load_bert_scorer():
    """Load BERTScore function."""
    try:
        from bert_score import score as bert_score_fn
        return bert_score_fn
    except ImportError:
        print("Error: bert-score not installed. Run: pip install bert-score")
        sys.exit(1)


def score_prompt_against_keypoints(
    prompt: str,
    key_points: List[str],
    bert_score_fn
) -> List[float]:
    """
    Score a prompt against all key points using BERTScore.
    Returns a list of F1 scores for each key point.
    """
    # Repeat the prompt for each key point
    cands = [prompt] * len(key_points)
    refs = key_points
    
    P, R, F1 = bert_score_fn(
        cands=cands,
        refs=refs,
        model_type=BERT_MODEL,
        verbose=False,
    )
    
    # Convert tensors to list of floats
    scores = [float(score.item()) for score in F1]  # Use .item() for PyTorch tensors
    return scores


def load_key_points(key_points_file: str = "repo_function.json") -> List[str]:
    """Load key points from JSON file."""
    with open(key_points_file, "r") as f:
        return json.load(f)


# ── ChromaDB Queries ──────────────────────────────────────────────────────────

def get_chroma_collection():
    """Get ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        return client.get_collection(CHROMA_COLLECTION)
    except ValueError:
        print(f"Error: Collection '{CHROMA_COLLECTION}' not found.")
        print("Run: python3 kb/vector/embeddings.py --reset")
        sys.exit(1)


def get_files_for_keypoint(
    collection, 
    key_point: str, 
    top_n: int = 2
) -> Tuple[List[str], List[Dict]]:
    """
    Get the top N files for a specific key point from ChromaDB.
    Uses metadata filtering to find files with matching top_kp.
    """
    print(f"\n  📂 Getting top {top_n} files for key point: '{key_point[:80]}...'")
    
    # Get all files from the collection
    results = collection.get(
        include=["documents", "metadatas"]
    )
    
    # Filter files that match the key point
    matching_files = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        if meta.get("top_kp") == key_point:
            matching_files.append({
                "doc": doc,
                "meta": meta
            })
    
    if not matching_files:
        print(f"  No files found for this key point.")
        return [], []
    
    # Sort by top_score (higher is better)
    matching_files.sort(key=lambda x: x["meta"].get("top_score", 0), reverse=True)
    
    # Take top N
    top_files = matching_files[:top_n]
    
    files = []
    metadata_list = []
    
    for i, item in enumerate(top_files, 1):
        meta = item["meta"]
        doc = item["doc"]
        
        files.append(meta["relative"])
        metadata_list.append({
            "relative": meta["relative"],
            "name": meta["name"],
            "similarity": meta.get("top_score", 0),
            "rule_count": meta.get("rule_count", 0),
            "top_kp": meta.get("top_kp", ""),
            "top_score": meta.get("top_score", 0),
            "rules": doc.split("\n\n") if doc else []
        })
        
        print(f"    {i}. {meta['name']} (score: {meta.get('top_score', 0):.4f})")
        print(f"       Rules: {meta.get('rule_count', 0)}")
    
    return files, metadata_list


# ── Neo4j Queries ─────────────────────────────────────────────────────────────

def get_dependencies(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    Get all DEPENDS_ON dependencies for the given files.
    Returns: {file_path: [dependency1, dependency2, ...]}
    """
    if not file_paths:
        return {}
    
    print(f"\n  🔗 Querying Neo4j for dependencies...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (f:File)-[:DEPENDS_ON]->(dep:File)
                WHERE f.relative IN $file_paths
                RETURN f.relative AS source, COLLECT(dep.relative) AS dependencies
            """, file_paths=file_paths)
            
            dependencies = {}
            for record in result:
                source = record["source"]
                deps = record["dependencies"]
                dependencies[source] = deps
                print(f"    {Path(source).name} depends on {len(deps)} files")
                for dep in deps[:3]:
                    print(f"      → {Path(dep).name}")
                if len(deps) > 3:
                    print(f"      ... and {len(deps) - 3} more")
            
            return dependencies
    
    finally:
        driver.close()


def get_file_locations(file_paths: List[str]) -> Dict[str, Dict]:
    """
    Get directory locations for files.
    """
    if not file_paths:
        return {}
    
    print(f"\n  📁 Getting file locations...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (d:Directory)-[:CONTAINS]->(f:File)
                WHERE f.relative IN $file_paths
                RETURN f.relative AS file, d.relative AS directory, d.name AS dir_name
            """, file_paths=file_paths)
            
            locations = {}
            for record in result:
                locations[record["file"]] = {
                    "directory": record["directory"],
                    "dir_name": record["dir_name"]
                }
            
            return locations
    
    finally:
        driver.close()


# ── PostgreSQL Queries ────────────────────────────────────────────────────────

def get_business_rules(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    Get business rules for the given files from PostgreSQL.
    Returns: {file_path: [rule1, rule2, ...]}
    """
    if not file_paths:
        return {}
    
    print(f"\n  📊 Querying PostgreSQL for business rules...")
    
    conn = psycopg.connect(**DB_CONFIG)
    
    try:
        with conn.cursor() as cur:
            # Get file IDs and rules
            cur.execute("""
                SELECT 
                    f.file_path,
                    array_agg(br.rule_text ORDER BY br.id) AS rules
                FROM files f
                JOIN business_rules br ON br.file_id = f.id
                WHERE f.file_path = ANY(%s)
                GROUP BY f.file_path
            """, (file_paths,))
            
            rules_dict = {}
            for row in cur.fetchall():
                file_path, rules = row
                rules_dict[file_path] = rules if rules else []
                print(f"    {Path(file_path).name}: {len(rules)} rules")
            
            return rules_dict
    
    finally:
        conn.close()


# ── Context Assembly ─────────────────────────────────────────────────────────

def assemble_context(
    query: str,
    key_point_scores: List[Tuple[str, float]],
    chroma_results: List[Dict],
    dependencies: Dict[str, List[str]],
    rules: Dict[str, List[str]],
    locations: Dict[str, Dict]
) -> Dict:
    """Assemble all retrieved context into a structured format."""
    
    context = {
        "query": query,
        "key_point_scores": [
            {"key_point": kp, "score": score} 
            for kp, score in key_point_scores
        ],
        "best_key_point": key_point_scores[0] if key_point_scores else None,
        "relevant_files": [],
        "all_dependencies": set(),
        "all_rules": [],
        "summary": {}
    }
    
    # Process each file
    for file_meta in chroma_results:
        file_path = file_meta["relative"]
        file_name = file_meta["name"]
        
        file_context = {
            "path": file_path,
            "name": file_name,
            "similarity": file_meta["similarity"],
            "top_key_point": file_meta["top_kp"],
            "location": locations.get(file_path, {}),
            "dependencies": dependencies.get(file_path, []),
            "business_rules": rules.get(file_path, []),
        }
        
        context["relevant_files"].append(file_context)
        
        # Collect all dependencies
        context["all_dependencies"].update(dependencies.get(file_path, []))
        
        # Collect all rules
        context["all_rules"].extend(rules.get(file_path, []))
    
    # Convert sets to lists for serialization
    context["all_dependencies"] = list(context["all_dependencies"])
    
    # Add summary
    context["summary"] = {
        "total_files": len(chroma_results),
        "total_dependencies": len(context["all_dependencies"]),
        "total_rules": len(context["all_rules"]),
        "best_key_point": key_point_scores[0][0] if key_point_scores else None,
        "best_key_point_score": key_point_scores[0][1] if key_point_scores else None,
    }
    
    return context


# ── Output Formatting ────────────────────────────────────────────────────────

def print_context(context: Dict):
    """Pretty print the assembled context."""
    
    print("\n" + "=" * 80)
    print(f"  📝 CONTEXT FOR: {context['query']}")
    print("=" * 80)
    
    # Show top key points
    print(f"\n  🎯 Top 5 Key Points:")
    key_point_scores = context.get('key_point_scores', [])
    for i, item in enumerate(key_point_scores[:5], 1):
        kp = item.get('key_point', 'Unknown')
        score = item.get('score', 0.0)
        # Ensure score is a float
        try:
            score_float = float(score)
        except (ValueError, TypeError):
            score_float = 0.0
        print(f"    {i}. {kp[:70]}... (score: {score_float:.4f})")
    
    # Summary
    summary = context.get('summary', {})
    print(f"\n  📊 Summary:")
    if summary.get('best_key_point'):
        print(f"    • Best key point: {summary['best_key_point'][:70]}...")
        print(f"    • Best score: {summary.get('best_key_point_score', 0):.4f}")
    print(f"    • Relevant files: {summary.get('total_files', 0)}")
    print(f"    • Dependencies: {summary.get('total_dependencies', 0)}")
    print(f"    • Business rules: {summary.get('total_rules', 0)}")
    
    # Files and their context
    relevant_files = context.get('relevant_files', [])
    for i, file_ctx in enumerate(relevant_files, 1):
        print(f"\n  ── File {i}: {file_ctx.get('name', 'Unknown')} ──")
        print(f"    Path: {file_ctx.get('path', 'Unknown')}")
        print(f"    ChromaDB Score: {file_ctx.get('similarity', 0):.4f}")
        
        location = file_ctx.get('location', {})
        if location:
            print(f"    Directory: {location.get('dir_name', 'unknown')}")
        
        # Business rules
        rules = file_ctx.get('business_rules', [])
        if rules:
            print(f"\n    📋 Business Rules ({len(rules)}):")
            for j, rule in enumerate(rules[:3], 1):
                print(f"      {j}. {rule[:150]}...")
            if len(rules) > 3:
                print(f"      ... and {len(rules) - 3} more")
        
        # Dependencies
        deps = file_ctx.get('dependencies', [])
        if deps:
            print(f"\n    🔗 Dependencies ({len(deps)}):")
            for dep in deps[:3]:
                print(f"      → {Path(dep).name}")
            if len(deps) > 3:
                print(f"      ... and {len(deps) - 3} more")
    
    # All dependencies summary
    all_deps = context.get('all_dependencies', [])
    if all_deps:
        print(f"\n  📦 All dependencies ({len(all_deps)} total):")
        for dep in all_deps[:5]:
            print(f"    • {Path(dep).name}")
        if len(all_deps) > 5:
            print(f"    ... and {len(all_deps) - 5} more")
    
    # All business rules summary
    all_rules = context.get('all_rules', [])
    if all_rules:
        print(f"\n  📋 All business rules ({len(all_rules)} total):")
        for rule in all_rules[:5]:
            print(f"    • {rule[:120]}...")
        if len(all_rules) > 5:
            print(f"    ... and {len(all_rules) - 5} more")
    
    print("\n" + "=" * 80)


def save_context(context: Dict, output_file: str = "context_output.json"):
    """Save context to JSON file."""
    with open(output_file, "w") as f:
        json.dump(context, f, indent=2, default=str)
    print(f"\n  💾 Context saved to: {output_file}")


# ── Main Query Function ──────────────────────────────────────────────────────

def query_context(prompt: str, top_n: int = 2, save: bool = False):
    """Main function to query all databases and assemble context."""
    
    print(f"\n  🚀 Starting context retrieval for: '{prompt}'")
    print("  " + "=" * 80)
    
    # Step 1: Load key points
    print("\n  📖 Loading key points...")
    try:
        key_points = load_key_points()
        print(f"    Loaded {len(key_points)} key points")
    except FileNotFoundError:
        print("  Error: repo_function.json not found. Run synthesize.py first.")
        return None
    
    # Step 2: Score prompt against key points using BERTScore
    print(f"\n  📊 Scoring prompt against key points using BERTScore...")
    bert_score_fn = load_bert_scorer()
    
    # Score the prompt against all key points
    scores = score_prompt_against_keypoints(prompt, key_points, bert_score_fn)
    
    # Pair key points with their scores and sort
    key_point_scores = list(zip(key_points, scores))
    key_point_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Display top 5 scores
    print(f"\n  🎯 Top 5 matching key points:")
    for i, (kp, score) in enumerate(key_point_scores[:5], 1):
        print(f"    {i}. {kp[:70]}... (score: {score:.4f})")
    
    if not key_point_scores or key_point_scores[0][1] < 0.3:
        print("\n  ⚠️  No strong matches found. Try a different query.")
        return None
    
    # Step 3: Get ChromaDB collection
    collection = get_chroma_collection()
    
    # Step 4: Get files for the best matching key point
    best_kp, best_score = key_point_scores[0]
    print(f"\n  ✅ Best match: '{best_kp[:80]}...' (score: {best_score:.4f})")
    
    file_paths, chroma_results = get_files_for_keypoint(
        collection, 
        best_kp, 
        top_n
    )
    
    if not file_paths:
        print("  No files found for this key point.")
        return None
    
    # Step 5: Get dependencies from Neo4j
    dependencies = get_dependencies(file_paths)
    
    # Step 6: Get locations from Neo4j
    locations = get_file_locations(file_paths)
    
    # Step 7: Get business rules from PostgreSQL
    rules = get_business_rules(file_paths)
    
    # Step 8: Assemble context
    context = assemble_context(
        prompt, 
        key_point_scores[:10],  # Keep top 10 for context
        chroma_results, 
        dependencies, 
        rules, 
        locations
    )
    
    # Step 9: Display and save
    print_context(context)
    
    if save:
        save_context(context)
    
    return context


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve context from all databases using BERTScore matching"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Query prompt (e.g., 'How does user authentication work?')"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=2,
        help="Number of top files to retrieve (default: 2)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save context to context_output.json"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        print("\n  Interactive Context Retrieval (using BERTScore)")
        print("  " + "=" * 80)
        print("  Enter prompts to query. Type 'quit' to exit.")
        print("  " + "=" * 80)
        
        while True:
            try:
                prompt = input("\n  🔍 Prompt: ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ["quit", "exit", "q"]:
                    break
                
                query_context(prompt, args.top, args.save)
            except KeyboardInterrupt:
                print("\n  Exiting...")
                break
    elif args.prompt:
        query_context(args.prompt, args.top, args.save)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()