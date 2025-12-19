# The optional values for the method argument are 'anchor-token', 'normal', 'kvcache', and 'anchor-thought'.
method="normal"
tokenizer_path="/mnt/jinbo/RLRM/model/Qwen/Qwen2.5-0.5B-Instruct"
comp_config="configs/LightThinker/qwen/v1.json"
model_type="qwen"
dataset="bbh"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
cache_size=1024
folder="1.5_wo_pretrain"
ckpt=5220
# file1="inference_results/${folder}/${dataset}/${ckpt}/1-4inf_qwen2.5_0.5b_tok_0.5b.jsonl"
# file2="inference_results/${folder}/${dataset}/${ckpt}/2-4inf_qwen2.5_0.5b_tok_0.5b.jsonl"
# file3="inference_results/${folder}/${dataset}/${ckpt}/3-4inf_qwen2.5_0.5b_tok_0.5b.jsonl"
# file4="inference_results/${folder}/${dataset}/${ckpt}/4-4inf_qwen2.5_0.5b_tok_0.5b.jsonl"
file1="/mnt/jinbo/RLRM/sglang_infer/results/bbh_result.jsonl"
python evaluation/eval_file.py \
  --method $method \
  --tokenizer_path $tokenizer_path \
  --comp_config $comp_config \
  --model_type $model_type \
  --dataset $dataset \
  --files $file1 $file2 $file3 $file4 \
  --cache_size $cache_size \
  --bos_token $bos_token \
  --eos_token $eos_token 
  # --interaction 