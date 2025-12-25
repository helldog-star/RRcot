import json
import jsonlines
from typing import *
import time
import regex as re
from dataset_reader import Reader, DATASET_PATH, _print, read_json

class MMLUCOTReader(Reader):

    file_path = DATASET_PATH['mmlu']
    
    def __init__(self):
        """
        {
            "file_name": [
                {"meta_data": [], "question": "", "answer": "D", "domain": ""}
            ]
        }
        """
        self.meta_db = read_json(MMLUCOTReader.file_path)

        self.data_list:List = list()
        for key in self.meta_db:
            for item in self.meta_db[key]:
                self.data_list.append(item)
                
    def get_prompt(self, idx:int) -> str:
        return "Please select the option that best answers the question. Return your final response within \\boxed{}.\nHere are the Question:\n" + self.data_list[idx]['question']

    def compare_answer(self, model_answer:str, gt_answer:str, idx:int) -> Tuple[bool, str]:
        if model_answer == "error" or model_answer == "" or model_answer == None:
            left_part = model_answer
            right_part = gt_answer
            return False, f"`{left_part}` <=> `{right_part}`"
        if model_answer[0].lower() in ['a', 'b', 'c', 'd']:
            left_part = model_answer[0].lower()
            right_part = gt_answer.lower()
            return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
        else:
            offset = ord(gt_answer)-ord('A')
            complete_answer = f"{gt_answer}. {self.data_list[idx]['choices_list'][offset]}"
            left_part = model_answer.lower().strip()
            right_part = complete_answer.lower().strip()
            return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
    
    def get_answer(self, idx: int) -> str:
        return self.data_list[idx]['answer']

    def get_system_prompt(self) -> str:
        return "Below is a question. Please think through it step by step, and then provide the final answer. If options are provided, please select the correct one.\n## Output format:\nUse “<THOUGHT>...</THOUGHT>” to outline your reasoning process, and enclose the final answer in ‘\\boxed{}‘.\n\n## Example 1:\nQuestion:\nWhat is 2 + 3?\nOutput:\n<THOUGHT>First, I recognize that this is a simple addition problem. Adding 2 and 3 together gives 5.</THOUGHT>\nTherefore, the final answer is \\boxed{5}.\n\n## Example 2:\nQuestion:\nWhat is 2 + 3?\nA. 4\nB. 5\nC. 10\n\nOutput:\n<THOUGHT>First, I recognize that this is a simple addition problem. Adding 2 and 3 together gives 5.</THOUGHT>\nTherefore, the final answer is \\boxed{B}."

class BBHCOTReader(Reader):

    file_path = DATASET_PATH['bbh']
    
    def __init__(self):
        """
        {
            "domain": [
                {
                    "meta_data": {
                        "input": "",
                        "target": "",
                        # `structured` is an optional choice
                        "structured": {
                            "question": "",
                            "choices": [
                                "",
                                "",

                            ],
                            "answer_flag": "", # A B C D E
                            "answer_content": ""
                        }
                    },
                    "question": "",
                    "answer": "",
                    "domain": ""
                },
            ]
        }
        """
        self.meta_db = read_json(BBHCOTReader.file_path)
        self.data_list:List = list()
        for key in self.meta_db:
            for item in self.meta_db[key]:
                self.data_list.append(item)
    
    def is_multiple_choices_question(self, idx) -> bool:
        return "structured" in self.data_list[idx]["meta_data"]

    def get_prompt(self, idx:int) -> str:
        return "Return your final response within \\boxed{}. If options are provided, please select the correct one. " + self.data_list[idx]['question']

    def compare_answer(self, model_answer:str, gt_answer:str, idx:int) -> Tuple[bool, str]:
        is_multi_choices:bool = self.is_multiple_choices_question(idx)
        if model_answer == "error":
            left_part = model_answer
            right_part = gt_answer
            return False, f"`{left_part}` <=> `{right_part}`"
        if not is_multi_choices:
            # is not a multiple-choice question
            left_part = model_answer.lower().strip()
            right_part = gt_answer.lower().strip()
            return left_part == right_part , f"`{left_part}` <=> `{right_part}`"
        else:
            if model_answer[0].lower() in [chr(ord('A')+i) for i in range(len(self.data_list[idx]["meta_data"]['structured']['choices']))]:
                left_part = model_answer[0].lower()
                right_part = gt_answer.lower()
                return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
            else:
                offset = ord(gt_answer) - ord('A')
                complete_answer = f"{gt_answer}. {self.data_list[idx]['meta_data']['structured']['choices'][offset]}"
                left_part = model_answer.lower().strip()
                right_part = complete_answer.lower().strip()
                return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
    
    def get_answer(self, idx: int) -> str:
        return self.data_list[idx]['answer']

    def get_system_prompt(self) -> str:
        return "Below is a question. Please think through it step by step, and then provide the final answer. If options are provided, please select the correct one.\n## Output format:\nUse “<THOUGHT>...</THOUGHT>” to outline your reasoning process, and enclose the final answer in ‘\\boxed{}‘.\n\n## Example 1:\nQuestion:\nWhat is 2 + 3?\nOutput:\n<THOUGHT>First, I recognize that this is a simple addition problem. Adding 2 and 3 together gives 5.</THOUGHT>\nTherefore, the final answer is \\boxed{5}.\n\n## Example 2:\nQuestion:\nWhat is 2 + 3?\nA. 4\nB. 5\nC. 10\n\nOutput:\n<THOUGHT>First, I recognize that this is a simple addition problem. Adding 2 and 3 together gives 5.</THOUGHT>\nTherefore, the final answer is \\boxed{B}."

class GSM8KCOTReader(Reader):

    file_path = DATASET_PATH['gsm8k']

    def __init__(self):
        """
        {
            "math": [
                {
                    "meta_data": {
                        "question": "",
                        "answer": ""
                    },
                    "question": "",
                    "answer": "",
                }
            ]
        }
        """
        self.meta_db = read_json(GSM8KCOTReader.file_path)
        self.data_list:List = list()
        for key in self.meta_db:
            for item in self.meta_db[key]:
                self.data_list.append(item)

    def get_prompt(self, idx:int) -> str:
        return "Return your final response within \\boxed{}. " + self.data_list[idx]['question']

    def compare_answer(self, model_answer:str, gt_answer:str, idx:int) -> bool:
        if model_answer == "error":
            left_part = model_answer
            right_part = gt_answer
            return False,  f"`{left_part}` <=> `{right_part}`"
        else:
            match = re.findall(r'\\boxed{(.*?)}', model_answer)
            if match:
                left_part = match[-1].strip().lower()
            else:
                left_part = model_answer.strip().lower()
            right_part = gt_answer.strip().lower()
            left_part = left_part.replace(",", "")
            right_part = right_part.replace(",", "")
            return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
        
    def get_answer(self, idx: int) -> str:
        return self.data_list[idx]['answer']

    def get_system_prompt(self) -> str:
        return "Below is a question. Please think through it step by step, and then provide the final answer. If options are provided, please select the correct one.\n## Output format:\nUse “<THOUGHT>...</THOUGHT>” to outline your reasoning process, and enclose the final answer in ‘\\boxed{}‘.\n\n## Example 1:\nQuestion:\nWhat is 2 + 3?\nOutput:\n<THOUGHT>First, I recognize that this is a simple addition problem. Adding 2 and 3 together gives 5.</THOUGHT>\nTherefore, the final answer is \\boxed{5}.\n\n## Example 2:\nQuestion:\nWhat is 2 + 3?\nA. 4\nB. 5\nC. 10\n\nOutput:\n<THOUGHT>First, I recognize that this is a simple addition problem. Adding 2 and 3 together gives 5.</THOUGHT>\nTherefore, the final answer is \\boxed{B}."

class GPQACOTReader(Reader):
    
    file_path = DATASET_PATH['gpqa']

    def __init__(self):
        """
        {
            "math": [
                {
                    "meta_data": [],
                    "question": "",
                    "answer": "",           
                    "choice_list": [
                        "", "", "", ""
                    ],
                    "answer_content": "",   
                    "pure_question": ""
                }
            ]
        }
        """
        self.meta_db = read_json(GPQACOTReader.file_path)
        self.data_list:List = list()
        for key in self.meta_db:
            for item in self.meta_db[key]:
                self.data_list.append(item)

    def get_prompt(self, idx:int) -> str:
        return "Given a question, please select the option that best answers it. Return your final response within \\boxed{}. " + self.data_list[idx]['question']

    def compare_answer(self, model_answer:str, gt_answer:str, idx:int) -> Tuple[bool, str]:
        if model_answer == "error":
            left_part = model_answer
            right_part = gt_answer
            return False, f"`{left_part}` <=> `{right_part}`"
        if model_answer[0].lower() in ['a', 'b', 'c', 'd']:
            left_part = model_answer[0].lower()
            right_part = gt_answer.lower()
            return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
        else:
            offset = ord(gt_answer)-ord('A')
            complete_answer = f"{gt_answer}. {self.data_list[idx]['choices_list'][offset]}"
            left_part = model_answer.lower().strip()
            right_part = complete_answer.lower().strip()
            return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
    
    def get_answer(self, idx: int) -> str:
        return str(self.data_list[idx]['answer'])

    def get_system_prompt(self) -> str:
        return "Below is a question. Please think through it step by step, and then provide the final answer. If options are provided, please select the correct one.\n## Output format:\nUse “<THOUGHT>...</THOUGHT>” to outline your reasoning process, and enclose the final answer in ‘\\boxed{}‘.\n\n## Example 1:\nQuestion:\nWhat is 2 + 3?\nOutput:\n<THOUGHT>First, I recognize that this is a simple addition problem. Adding 2 and 3 together gives 5.</THOUGHT>\nTherefore, the final answer is \\boxed{5}.\n\n## Example 2:\nQuestion:\nWhat is 2 + 3?\nA. 4\nB. 5\nC. 10\n\nOutput:\n<THOUGHT>First, I recognize that this is a simple addition problem. Adding 2 and 3 together gives 5.</THOUGHT>\nTherefore, the final answer is \\boxed{B}."
