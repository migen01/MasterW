from ctypes import util
from os import fsync
import time
from openai import OpenAI
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



from datasets import load_dataset
squad_dataset = load_dataset('squad', split='validation')[0]

#top_5_entries = squad_dataset.select(range(5))

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class LogicModule:
    def __init__(self, rules):
        self.rules = rules
    
    def apply_rules(self,data):
        #example
        if data['input1'] and data['input2']:
            return True
        return False
    
input_size = 10
hidden_size = 5
output_size = 2
neural_net= SimpleNN(input_size, hidden_size , output_size)
rules = {'rule1':'input1 AND input2'}
logic_module= LogicModule(rules)


client = OpenAI()

def generate_answer(promt):
     # Call the OpenAI API to get the response
    messages=[
            {"role":"system", "content": "You are a helpful assistent. Only one word answer"},
            {"role": "user","content": promt}
        ]
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=messages,
        max_tokens=10,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def query_database(question):
    db_answers = {
        "What is the capital of France?": "Paris",
        "What is 2+2?": "4"
    }
    return db_answers.get(question, "no answ")

def compare_answers(question):
    db_answer = query_database(question)
    llm_answer = generate_answer(question)

    print(f"Database Answer: {db_answer}")
    print(f"LLM Answer: {llm_answer}")
    
    if db_answer.lower() == llm_answer.lower():
        return True
    return False



class HybridModel:
    def __init__(self, neural_net, logic_module):
        self.neural_net = neural_net
        self.logic_module = logic_module

    def forward( self, x, logic_data):
        nn_output = self.neural_net(x)
        logic_output = self.logic_module.apply_rules(logic_data)
        combined_output = nn_output * logic_output
        return combined_output

input_tensor = torch.randn(1, input_size)
logic_data = {'input1': True, 'input2': False}

#example 
hybrid_model = HybridModel(neural_net, logic_module)
output = hybrid_model.forward(input_tensor, logic_data)
print(output)

question = "What is the capital of France?"
is_correct = compare_answers(question)

print(f"Are the answers consistent? {is_correct}")


def hybrid_output_f(question, hybrid_model, input_tensor, logic_data):
    llm_answer = generate_answer(question)
    hybrid_output = hybrid_model.forward(input_tensor, logic_data)
    print(f"LLM Answer:{llm_answer}")
    print(f"hybrid Answer:{hybrid_output}")

    return llm_answer, hybrid_output

llm_answer, hybrid_output = hybrid_output_f(question, hybrid_model, input_tensor, logic_data)
print(f"Refined Answer: {llm_answer}, Hybrid Output: {hybrid_output}")

