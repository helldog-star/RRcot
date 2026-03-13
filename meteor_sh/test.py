
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('/mnt/lxy/hf_models/Qwen2.5-1.5B-Instruct')


test_text = """<|im_start|>At three o'clock in the morning in the convenience store, the fluorescent lights made everything look very pale. 
Lin Sheng stood behind the counter of the Kodo Soup restaurant, watching the last customer walk out of the automatic door. As the door closed, there was a soft "ding-dong" sound, and then the entire restaurant was left with only the humming of the refrigerator. 
He glanced at the clock on the wall. There were still four hours before the shift change. 
The night shift at the convenience store is like this - most of the time it's completely empty. Sometimes only three or four people come throughout the entire night, and sometimes not even a single person shows up. Tonight was a relatively busy night, with a total of seven customers so far. 
The seventh one was the girl who arrived twenty minutes earlier. She was wearing pajamas, with a down jacket over them, and was wearing cotton slippers. She walked around the shelves three times and finally picked up a box of strawberry milk. When checking out, Lin Sheng noticed that her eyes were a little red and her hair was messy, as if she had just gotten out of bed.
"""
tokens_1 = tokenizer(test_text, add_special_tokens=True)


print(tokens_1)

test_text = """<|im_start|>At three o'clock in the morning in the convenience store, the fluorescent lights made everything look very pale. 
Lin Sheng stood behind the counter of the Kodo Soup restaurant, watching the last customer walk out of the automatic door. As the door closed, there was a soft "ding-dong" sound, and then the entire restaurant was left with only the humming of the refrigerator. 
He glanced at the clock on the wall. There were still four hours before the shift change. 
The night shift at the convenience store is like this - most of the time it's completely empty. Sometimes only three or four people come throughout the entire night, and sometimes not even a single person shows up. Tonight was a relatively busy night, with a total of seven customers so far. 
The seventh one was the girl who arrived twenty minutes earlier. She was wearing pajamas, with a down jacket over them, and was wearing cotton slippers. She walked around the shelves three times and finally picked up a box of strawberry milk. When checking out, Lin Sheng noticed that her eyes were a little red and her hair was messy, as if she had just gotten out of bed.
"""
tokens_2 = tokenizer(test_text, add_special_tokens=False)
print(tokens_2)



