from pydantic import BaseModel
from typing import Literal

class BusinessRule(BaseModel):
    #Class used to specify a business rule
    description: str
    category: Literal['EXPLICIT', 'IMPLICIT']
    filename: str

class RuleInventory(BaseModel):
    #Class used to store business rules
    rules: list[BusinessRule]

class BRScores(BaseModel):
    #Class used to store the scores and feedback for each respective business rule
    clarity: Literal[0, 1, 2, 3, 5, 6, 7, 8, 9, 10] 
    accuracy: Literal[0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
    coverage: Literal[0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
    overall: Literal[0, 1, 2, 3, 5, 6, 7, 8, 9, 10]