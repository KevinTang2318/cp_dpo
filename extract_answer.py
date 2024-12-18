import pandas as pd 


df = pd.read_json("v12_perform_pipeline_full_test3.json")

import json
import re

import pandas as pd
from sklearn.metrics import accuracy_score

query = {}
answer = {}
question_list = {}
for idx, content in df.iterrows():
    if content.dataset not in query.keys():
        query[content.dataset] = []
    if content.dataset not in answer.keys():
        answer[content.dataset] = []
    response = content.answer
    
    if content.question not in question_list:
        # print(content.answer)
        if "correct answer:" in content.answer.lower() :
            response = content.answer.lower().replace("correct answer: <your answer>", "").split("correct answer: ")[1].split(";")[0].replace(".", "")
        
        if content.dataset == "last_letter":
            response = response.replace("-", "").replace(",", "").replace(" ", "")
        
        if content.dataset == "aqua":
            response = response.split(")")[0]
        if content.dataset == "object_tracking":
            ground_truth = content.ground_truth.replace(".", "").lower()
        else:
            ground_truth = content.ground_truth.lower()
        
        query[content.dataset].append(ground_truth)
        answer[content.dataset].append(response.lower())
        question_list[(content.question)] = []

for key in query:
    print(key)
    print(accuracy_score(query[key], answer[key]))
    

