
# ==================== 推理参数（可直接在脚本内修改） ====================
# 直接运行: bash inference_je.sh
# 可选覆盖: bash inference_je.sh [model_tag] [repetition_penalty] [ckpt] [root_dir] [output_base_dir] [tokenizer_path]

# 默认参数（推荐在这里改）
model_tag="epl_apa_true"
repetition_penalty="1.1"
ckpt="5220"
root_dir="/mnt/jinbo/RLRM/RRcot"
output_base_dir="/mnt/jinbo/RLRM/RRcot/output"
tokenizer_path="/mnt/jinbo/RLRM/model/Qwen/Qwen2.5-7B-Instruct"
seman="false"  # 是否启用语义过滤（seman_memcot），true或false
seman_conf_threshold="0.0294493680766361"

# 可选: 使用命令行参数覆盖默认参数
if [ $# -gt 0 ]; then
    if [ $# -lt 6 ]; then
        echo "错误: 传参模式下缺少必需参数"
        echo "使用方法: $0 [model_tag] [repetition_penalty] [ckpt] [root_dir] [output_base_dir] [tokenizer_path] [seman可选] [seman_conf_threshold可选]"
        exit 1
    fi

    model_tag="$1"
    repetition_penalty="$2"
    ckpt="$3"
    root_dir="$4"
    output_base_dir="$5"
    tokenizer_path="$6"
    if [ $# -ge 7 ]; then
        seman="$7"
    fi
    if [ $# -ge 8 ]; then
        seman_conf_threshold="$8"
    fi
fi

# 根据model_tag自动调整use_EPL：vanilla和lightthinker为false，其余为true
if [ "$model_tag" = "vanilla" ] || [ "$model_tag" = "lightthinker" ]; then
    use_EPL="false"
else
    use_EPL="true"
fi

# 检查必需参数是否为空
if [ -z "$model_tag" ] || [ -z "$repetition_penalty" ] || [ -z "$ckpt" ] || [ -z "$root_dir" ] || [ -z "$output_base_dir" ] || [ -z "$tokenizer_path" ] || [ -z "$seman" ] || [ -z "$seman_conf_threshold" ]; then
    echo "错误: model_tag, repetition_penalty, ckpt, root_dir, output_base_dir, tokenizer_path, seman, seman_conf_threshold 不能为空"
    echo "model_tag: $model_tag"
    echo "repetition_penalty: $repetition_penalty"
    echo "ckpt: $ckpt"
    echo "root_dir: $root_dir"
    echo "output_base_dir: $output_base_dir"
    echo "tokenizer_path: $tokenizer_path"
    echo "seman: $seman"
    echo "seman_conf_threshold: $seman_conf_threshold"
    exit 1
fi

# 检查 root_dir 是否存在
if [ ! -d "$root_dir" ]; then
    echo "错误: root_dir 不存在: $root_dir"
    exit 1
fi

cd "$root_dir"

# 根据传入的超参数自动组合路径
output_path="${output_base_dir}/${model_tag}"
output_tag="${output_path}/inference"
model_path="${output_path}/train/checkpoint-${ckpt}"

# 检查模型路径是否存在
if [ ! -d "$model_path" ]; then
    echo "警告: 模型路径不存在: $model_path"
    echo "请确认 model_tag 和 ckpt 是否正确"
fi

export PYTHONPATH=$PYTHONPATH:$root_dir

model_short_tag="${model_tag}"

model_type="qwen"
# tokenizer_path 从命令行参数传入
bos_token="<|im_start|>"
eos_token="<|im_end|>"
compress_config="${root_dir}/configs/LightThinker/qwen/v1.json"

# `model_path` is an optional argument
# if you set the `model_path`, the arguments `ckpt` and `model_tag` will be ignored.
# see line 1460 of the code in LightThinker/inference.py for more details.
max_new_tokens=10240

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
echo "seman: ${seman}"
echo "seman_conf_threshold: ${seman_conf_threshold}"
echo "output_path: ${output_tag}"
echo "model_path: ${model_path}"
echo "Inference model: ${model_tag}..."

#用于设置总共几张卡和开多少进程
target_gpus=( 0 1 2 3 )
process_per_gpu=4
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
        CUDA_VISIBLE_DEVICES=$device nohup python "${root_dir}/LightThinker/inference.py" \
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
            --seman $seman \
            --seman_conf_threshold $seman_conf_threshold \
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