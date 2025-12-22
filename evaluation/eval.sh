# The optional values for the method argument are 'anchor-token', 'normal', 'kvcache', and 'anchor-thought'.
method="anchor-thought"
tokenizer_path="Qwen/Qwen2.5-7B-Instruct"
comp_config="configs/LightThinker/qwen/v1.json"
model_type="qwen"
dataset="gpqa"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
cache_size=1024
folder=""
ckpt=1045
file1="inference_results/${folder}/${dataset}/${ckpt}/1-4qwen_7b.jsonl"
file2="inference_results/${folder}/${dataset}/${ckpt}/2-4qwen_7b.jsonl"
file3="inference_results/${folder}/${dataset}/${ckpt}/3-4qwen_7b.jsonl"
file4="inference_results/${folder}/${dataset}/${ckpt}/4-4qwen_7b.jsonl"
python eval_file.py \
  --method $method \
  --tokenizer_path $tokenizer_path \
  --comp_config $comp_config \
  --model_type $model_type \
  --dataset $dataset \
  --files $file1 $file2 $file3 $file4 \
  --cache_size $cache_size \
  --bos_token $bos_token \
  --eos_token $eos_token \
#   --interaction 