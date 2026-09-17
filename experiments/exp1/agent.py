import litellm
import json
import os

from prompt import build_messages, PROMPTS

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_business_rules(model = 'gpt-4', message = 'Extract business rules'):
    response = litellm.completion()