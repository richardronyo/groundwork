import litellm
import json
import os
import csv

from prompt import build_messages, PROMPTS

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_business_rules(model: str, prompt: str, files: list[str]) -> list[dict]:
    business_rules = []
    for file in files:
        with open(file, 'r') as f:
            code = f.read()

        message = build_messages(prompt, file, code)

        response = litellm.completion(
            model = model,
            messages = message,
            temperature = 0
        )

        raw_text = response.choices[0].message.content
        rules = json.loads(raw_text)

        business_rules.extend(rules)

    return business_rules

if __name__ == "__main__":
    files = []
    for file in os.listdir('data/code'):
        files.append(f'data/code/{file}') 

    part1 = ['general', 'user_story', 'definition']
    model = 'gpt-4o-mini'

    for prompt_type in part1:
        prompt = PROMPTS[prompt_type]
        business_rules = get_business_rules(model, prompt, files)

        with open(f'outputs/{prompt_type}.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['file', 'business_rule'])
                writer.writeheader()
                writer.writerows(business_rules)