# ==================== 推理参数（可直接在脚本内修改） ====================
# 直接运行: bash sglang_inference.sh
# 可选覆盖: bash sglang_inference.sh [model_tag] [repetition_penalty] [ckpt] [root_dir] [output_base_dir] [conda_sh_path] [conda_env_name]

# 默认参数（推荐在这里改）
model_tag="vanilla"
repetition_penalty="1.1"
ckpt="5220"
root_dir="/mnt/jinbo/RLRM/RRcot"
output_base_dir="/mnt/jinbo/RLRM/RRcot/output"
conda_sh_path="/mnt/jinbo/miniconda3/etc/profile.d/conda.sh"
conda_env_name="sglang"

# 其他推理参数
datasets="mmlu gsm8k gpqa bbh"
batch_size=16
tp_size=4
cuda_visible_devices="4,5,6,7"
extend_name="inference"  # 与 inference.sh 的 output_tag 保持一致

# 可选: 使用命令行参数覆盖默认参数
if [ $# -gt 0 ]; then
    if [ $# -lt 7 ]; then
        echo "错误: 传参模式下缺少必需参数"
        echo "使用方法: $0 [model_tag] [repetition_penalty] [ckpt] [root_dir] [output_base_dir] [conda_sh_path] [conda_env_name]"
        exit 1
    fi

    model_tag="$1"
    repetition_penalty="$2"
    ckpt="$3"
    root_dir="$4"
    output_base_dir="$5"
    conda_sh_path="$6"
    conda_env_name="$7"
fi

# 检查必需参数是否为空
if [ -z "$model_tag" ] || [ -z "$repetition_penalty" ] || [ -z "$ckpt" ] || [ -z "$root_dir" ] || [ -z "$output_base_dir" ] || [ -z "$conda_sh_path" ] || [ -z "$conda_env_name" ]; then
    echo "错误: model_tag, repetition_penalty, ckpt, root_dir, output_base_dir, conda_sh_path, conda_env_name 不能为空"
    echo "model_tag: $model_tag"
    echo "repetition_penalty: $repetition_penalty"
    echo "ckpt: $ckpt"
    echo "root_dir: $root_dir"
    echo "output_base_dir: $output_base_dir"
    echo "conda_sh_path: $conda_sh_path"
    echo "conda_env_name: $conda_env_name"
    exit 1
fi

# 检查路径是否存在
if [ ! -d "$root_dir" ]; then
    echo "错误: root_dir 不存在: $root_dir"
    exit 1
fi

if [ ! -f "$conda_sh_path" ]; then
    echo "错误: conda_sh_path 不存在: $conda_sh_path"
    exit 1
fi

cd "$root_dir"

# 根据传入的超参数自动组合路径
output_path="${output_base_dir}/${model_tag}"
model_path="${output_path}/train/checkpoint-${ckpt}"

# 检查模型路径是否存在
if [ ! -d "$model_path" ]; then
    echo "警告: 模型路径不存在: $model_path"
    echo "请确认 model_tag 和 ckpt 是否正确"
fi

# 激活 conda 环境
source "$conda_sh_path"
conda activate "$conda_env_name"
echo "Using python: $(which python)"

# export PYTHONPATH="$root_dir:${PYTHONPATH}"

output_dir="${output_path}"

echo "model_tag: ${model_tag}"
echo "repetition_penalty: ${repetition_penalty}"
echo "datasets: ${datasets}"
echo "batch_size: ${batch_size}"
echo "tp_size: ${tp_size}"
echo "CUDA_VISIBLE_DEVICES: ${cuda_visible_devices}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "Inference model: ${model_tag} using sglang..."

CUDA_VISIBLE_DEVICES="$cuda_visible_devices" python "${root_dir}/LightThinker/sglang_inference.py" \
  --model_path $model_path \
  --datasets $datasets \
  --batch_size $batch_size \
  --output_dir $output_dir \
    --tp_size $tp_size \
  --extend_name $extend_name \
  --repetition_penalty $repetition_penalty
