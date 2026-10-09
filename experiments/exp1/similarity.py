import operator

from pydantic import BaseModel
from typing import Annotated, TypedDict
from langchain_litellm import ChatLiteLLM
from langgraph.graph import StateGraph, START, END
from prompt import SIMILARITY_PROMPT, SIMILARITY_REVIEW_PROMPT

MAX_SIM_ITERATION = 3
judge = ChatLiteLLM(model_name="gpt-4o-mini", temperature=0)

class Similarity(BaseModel):
    #Class used to store the similarity between two statements
    reason: str
    score: float

class SimState(TypedDict, total = False):
    statement_a: str
    statement_b: str
    score: float
    reason: str
    iteration: int
    history: Annotated[list[dict], operator.add]

def sim_scorer(state: SimState) -> dict:
    #Scores the pair, and from the second pass on, reviews the previous score
    prompt = f'{SIMILARITY_PROMPT}\nStatement A: {state["statement_a"]}\nStatement B: {state["statement_b"]}'
    if 'score' in state:
        prompt += f'\n\n{SIMILARITY_REVIEW_PROMPT}\nPrevious score: {state["score"]}\nPrevious reason: {state["reason"]}'
    result = judge.with_structured_output(Similarity).invoke(prompt)
    score = min(max(result.score, 0.0), 1.0)
    iteration = state.get('iteration', 0) + 1
    return {'score': score, 'reason': result.reason, 'iteration': iteration,
            'history': [{'iteration': iteration, 'score': score, 'reason': result.reason}]}

def sim_next(state: SimState) -> str:
    #Stop after MAX_SIM_ITERATION passes, or once the score stops changing
    history = state['history']
    settled = len(history) >= 2 and history[-1]['score'] == history[-2]['score']
    return END if settled or state['iteration'] >= MAX_SIM_ITERATION else 'Scorer'

def sim_loop():
    graph = StateGraph(SimState)
    graph.add_node('Scorer', sim_scorer)
    graph.add_edge(START, 'Scorer')
    graph.add_conditional_edges('Scorer', sim_next, ['Scorer', END])
    return graph.compile()

sim_app = sim_loop()

def similarity(statement_a: str, statement_b: str) -> float:
    #Returns the final similarity score, from 0 to 1, for two statements
    return sim_app.invoke({'statement_a': statement_a, 'statement_b': statement_b})['score']