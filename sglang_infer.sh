
model_path="/mnt/jinbo/RLRM/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
datasets="gpqa"
batch_size=8
output_dir="./sglang_inference_results"
extend_name="7b"

root_dir="./LightThinker"


  python "${root_dir}/sglang_inference.py" \
    --model_path $model_path \
    --datasets $datasets \
    --output_dir $output_dir \
    --batch_size $batch_size \
    --extend_name $extend_name