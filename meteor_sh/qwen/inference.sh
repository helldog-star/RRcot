
# ==================== 通过命令行传入必要超参数 ====================
# 使用方法: ./script.sh [model_tag] [repetition_penalty] [ckpt] [root_dir] [output_base_dir] [tokenizer_path]
# 示例: ./script.sh "lightthinker" "1.1" "1305" "./LightThinker" "/tmp/hx/rrcot" "/tmp/hx/Qwen/Qwen2.5-1.5B-Instruct"

# 检查必需参数（至少需要6个：model_tag, repetition_penalty, ckpt, root_dir, output_base_dir, tokenizer_path）
if [ $# -lt 6 ]; then
    echo "错误: 缺少必需的超参数"
    echo "使用方法: $0 [model_tag] [repetition_penalty] [ckpt] [root_dir] [output_base_dir] [tokenizer_path]"
    echo "  model_tag: 必需，模型标识（与训练时的init_tag一致）"
    echo "  repetition_penalty: 必需，重复惩罚系数"
    echo "  ckpt: 必需，检查点编号"
    echo "  root_dir: 必需，代码根目录"
    echo "  output_base_dir: 必需，输出基础目录"
    echo "  tokenizer_path: 必需，tokenizer路径"
    echo "示例: $0 \"lightthinker\" \"1.1\" \"1305\" \"/zhaorunsong/RRcot/LightThinker\" \"/tmp/hx/rrcot\" \"/tmp/hx/Qwen/Qwen2.5-1.5B-Instruct\""
    exit 1
fi

# 从命令行参数获取
model_tag="$1"
repetition_penalty="$2"
ckpt="$3"
root_dir="$4"
output_base_dir="$5"
tokenizer_path="$6"
compress_config="$7"

# 根据model_tag自动调整use_EPL：vanilla和lightthinker为false，其余为true
if [ "$model_tag" = "vanilla" ] || [ "$model_tag" = "lightthinker" ] || [ "$model_tag" = "distill-r1-7b" ]; then
    use_EPL="false"
else
    use_EPL="true"
fi

# 检查必需参数是否为空
if [ -z "$model_tag" ] || [ -z "$repetition_penalty" ] || [ -z "$ckpt" ] || [ -z "$root_dir" ] || [ -z "$output_base_dir" ] || [ -z "$tokenizer_path" ]; then
    echo "错误: model_tag, repetition_penalty, ckpt, root_dir, output_base_dir, tokenizer_path 不能为空"
    echo "model_tag: $model_tag"
    echo "repetition_penalty: $repetition_penalty"
    echo "ckpt: $ckpt"
    echo "root_dir: $root_dir"
    echo "output_base_dir: $output_base_dir"
    echo "tokenizer_path: $tokenizer_path"
    exit 1
fi

# 根据传入的超参数自动组合路径
output_path="${output_base_dir}/${model_tag}"
output_tag="${output_path}/inference"
model_path="${output_path}/train/checkpoint-${ckpt}"

# 检查模型路径是否存在
if [ ! -d "$model_path" ]; then
    echo "警告: 模型路径不存在: $model_path"
    echo "请确认 model_tag 和 ckpt 是否正确"
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)

model_short_tag="${model_tag}"

model_type="qwen"
# tokenizer_path, model_path, train_data_path 从命令行参数传入
bos_token="<|im_start|>"
eos_token="<|im_end|>"
# compress_config="./configs/LightThinker/qwen/v1.json"

# `model_path` is an optional argument
# if you set the `model_path`, the arguments `ckpt` and `model_tag` will be ignored.
# see line 1460 of the code in LightThinker/inference.py for more details.
max_new_tokens=1024

prefix=""
diagonal="false"
see_current="false"
compress_prompt="false"
rolling_rope="false"
bi_directional="false"
exclude_continue="false"
output_compress_instruction="None"
prefill_compress="false"
update_attention_method="local"


# check "inference_log" 
if [ ! -d "${output_tag}/inference_log" ]; then
    echo "Creating ${output_tag}/inference_log directory..."
    mkdir -p "${output_tag}/inference_log"
fi

subfolders=("true_true" "true_false" "false_false" "false_true")
for subfolder in "${subfolders[@]}"; do
    folder_path="${output_tag}/inference_log/${subfolder}"
    if [ ! -d "$folder_path" ]; then
        echo "Creating $folder_path directory..."
        mkdir -p "$folder_path"
    fi
done

echo "model_tag: ${model_tag}"
echo "repetition_penalty: ${repetition_penalty}"
echo "use_EPL: ${use_EPL}"
echo "output_path: ${output_tag}"
echo "model_path: ${model_path}"
echo "Inference model: ${model_tag}..."

#用于设置总共几张卡和开多少进程
target_gpus=( 0 5 6 7)
process_per_gpu=2
gpu_count=${#target_gpus[@]}
# 自动计算总切片数 (假如用了2张卡，每张3进程，split_size就是6)
split_size=$((gpu_count * process_per_gpu))

logical_id=0
for device in "${target_gpus[@]}"
do
    # 计算当前显卡负责的 "0-based" 索引范围 (例如 0,1,2)
    start_index_0based=$((logical_id * process_per_gpu))
    end_index_0based=$((start_index_0based + process_per_gpu - 1))
    echo ">>> Launching on Physical GPU ${device}" 
    for ((idx=start_index_0based; idx<=end_index_0based; idx++))
    do
        real_index=$((idx + 1))
        
        echo "    Starting task index ${real_index}/${split_size}..."

        # 评测EPL训练模型时 --EPL=True 
        CUDA_VISIBLE_DEVICES=$device nohup python "${root_dir}/inference.py" \
            --model_tag $model_tag \
            --model_short_tag $model_short_tag \
            --ckpt $ckpt \
            --tokenizer_path $tokenizer_path \
            --compress_config $compress_config \
            --max_new_tokens $max_new_tokens \
            --repetition_penalty $repetition_penalty \
            --output_tag $output_tag \
            --model_type $model_type \
            --bos_token $bos_token \
            --eos_token $eos_token \
            --rolling_rope $rolling_rope \
            --diagonal $diagonal \
            --bi_directional $bi_directional \
            --see_current $see_current \
            --exclude_continue $exclude_continue \
            --output_compress_instruction $output_compress_instruction \
            --prefill_compress $prefill_compress \
            --compress_prompt $compress_prompt \
            --update_attention_method $update_attention_method \
            --split_size $split_size \
            --use_EPL $use_EPL \
            --model_path $model_path \
            --index $real_index > "${output_tag}/inference_log/${rolling_rope}_${compress_prompt}/${real_index}${prefix}_${model_short_tag}_${ckpt}.txt" 2>&1 &
        
        sleep 5
    done
    ((logical_id++))
done

echo ""
echo "=========================================="
echo "All processes launched. Waiting for completion..."
echo "Started at: $(date)"
echo "=========================================="

wait  # 等待所有后台进程（&）完成

echo ""
echo "=========================================="
echo "All processes completed!"
echo "Finished at: $(date)"
echo "=========================================="