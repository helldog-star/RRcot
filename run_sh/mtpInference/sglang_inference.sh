# ==================== 通过命令行传入必要超参数 ====================
# 使用方法: ./script.sh [model_tag] [repetition_penalty] [ckpt] [root_dir] [output_base_dir] [conda_sh_path] [conda_env_name]
# 示例: ./script.sh "vanilla" "1.1" "1305" "./LightThinker" "/tmp/hx/rrcot" "/mnt/user/anaconda3/etc/profile.d/conda.sh" "niah"
# 注意: 此脚本用于 vanilla 模型的推理，使用 sglang

# 检查必需参数（至少需要7个：model_tag, repetition_penalty, ckpt, root_dir, output_base_dir, conda_sh_path, conda_env_name）
if [ $# -lt 7 ]; then
    echo "错误: 缺少必需的超参数"
    echo "使用方法: $0 [model_tag] [repetition_penalty] [ckpt] [root_dir] [output_base_dir] [conda_sh_path] [conda_env_name]"
    echo "  model_tag: 必需，模型标识（与训练时的init_tag一致）"
    echo "  repetition_penalty: 必需，重复惩罚系数"
    echo "  ckpt: 必需，检查点编号"
    echo "  root_dir: 必需，代码根目录"
    echo "  output_base_dir: 必需，输出基础目录"
    echo "  conda_sh_path: 必需，conda初始化脚本路径"
    echo "  conda_env_name: 必需，conda环境名称"
    echo "示例: $0 \"vanilla\" \"1.1\" \"1305\" \"/zhaorunsong/RRcot/LightThinker\" \"/tmp/hx/rrcot\" \"/mnt/user/anaconda3/etc/profile.d/conda.sh\" \"niah\""
    exit 1
fi

# 从命令行参数获取
model_tag="$1"
repetition_penalty="$2"
ckpt="$3"
root_dir="$4"
output_base_dir="$5"
conda_sh_path="$6"
conda_env_name="$7"

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

# export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 设置默认参数
datasets="mmlu gsm8k gpqa bbh"
batch_size=16
output_dir="${output_path}"
extend_name="inference"  # 与 inference.sh 的 output_tag 保持一致

echo "model_tag: ${model_tag}"
echo "repetition_penalty: ${repetition_penalty}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "Inference model: ${model_tag} using sglang..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python "${root_dir}/sglang_inference.py" \
  --model_path $model_path \
  --datasets $datasets \
  --batch_size $batch_size \
  --output_dir $output_dir \
  --tp_size 2 \
  --extend_name $extend_name \
  --repetition_penalty $repetition_penalty
