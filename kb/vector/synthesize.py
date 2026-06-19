import json
from pathlib import Path

from extract_business_rules import (
    get_client,
    synthesize_repo_function
)

rules_file = Path("business_rules.json")

with open(rules_file) as f:
    business_rules = json.load(f)

print(business_rules)
client = get_client()

repo_function = synthesize_repo_function(
    business_rules,
    client
)

with open("repo_function.json", "w") as f:
    json.dump(repo_function, f, indent=2)

print(f"Generated {len(repo_function)} capabilities")