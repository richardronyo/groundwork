import json
import operator
import pandas as pd
import os

from langchain_litellm import ChatLiteLLM
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from components import RuleInventory, BRScores
from prompt import COBREX_BR, USER_STORY_BR, SCORE_PROMPT, REFINER_PROMPT
from pathlib import Path
model = ChatLiteLLM(model_name="gpt-4o-mini")
MAX_BR_ITERATION = 3

class State(TypedDict, total = False):
    filename: str
    source_code: str
    br_prompt: str
    rules: list[dict]
    scores: dict
    iteration: int
    history: Annotated[list[dict], operator.add]

def extractor(state: State) -> dict:
    prompt = f'{state["br_prompt"]}\nFilename: {state["filename"]}\nSource Code: {state["source_code"]}'
    rules = model.with_structured_output(RuleInventory).invoke(prompt).model_dump()['rules']
    return {'rules': rules, 'iteration': 0}

def scorer(state: State) -> dict:
    prompt = f'{SCORE_PROMPT}\nFilename: {state["filename"]}\nSource Code: {state["source_code"]}\nBusiness Rules: {json.dumps(state["rules"], indent=2)}'
    scores = model.with_structured_output(BRScores).invoke(prompt).model_dump()
    return {'scores': scores, 'history': [{'iteration': state['iteration'], 'rules': state['rules'], 'scores': scores}]}

def refiner(state: State) -> dict:
    prompt = (f'{REFINER_PROMPT}\nFilename: {state["filename"]}\nSource Code: {state["source_code"]}\n'
              f'Business Rules: {json.dumps(state["rules"], indent=2)}\nScores: {json.dumps(state["scores"], indent=2)}')
    rules = model.with_structured_output(RuleInventory).invoke(prompt).model_dump()['rules']
    return {'rules': rules, 'iteration': state['iteration'] + 1}

def next_step(state: State) -> dict:
    return END if state['iteration'] >= MAX_BR_ITERATION else 'Refiner'

def br_loop():
    graph = StateGraph(State)
    graph.add_node('Extractor', extractor)
    graph.add_node('Scorer', scorer)
    graph.add_node('Refiner', refiner)

    graph.add_edge(START, 'Extractor')
    graph.add_edge('Extractor', 'Scorer')
    graph.add_conditional_edges('Scorer', next_step, ['Refiner', END])
    graph.add_edge('Refiner', 'Scorer')

    return graph.compile()


def save_history(runs: dict[str, list[dict]], folder: str = 'outputs/user_story') -> None:
    #Writes one CSV per iteration, holding the rules and scores from every source file
    frames = [pd.DataFrame(entry['rules']).assign(filename=filename, iteration=entry['iteration'], **entry['scores'])
              for filename, history in runs.items() for entry in history]
    combined = pd.concat(frames, ignore_index=True)
    for iteration, df in combined.groupby('iteration'):
        df.drop(columns='iteration').to_csv(f'{folder}/br{iteration}.csv', index=False)

if __name__ == '__main__':
    app = br_loop()
    runs = {}

    for path in sorted(Path('data/code').rglob('*.py')):
        filename = str(path.relative_to('data/code'))
        print(filename)

        result = app.invoke({
            'filename': filename,
            'source_code': path.read_text(),
            'br_prompt': USER_STORY_BR,
        })

        runs[filename] = result['history']
        save_history(runs)      #Saved after every file, so a crash keeps the finished ones