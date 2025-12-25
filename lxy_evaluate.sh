#!/bin/bash

# 默认参数
DEFAULT_METHOD="anchor-thought"
DEFAULT_TOKENIZER_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen2.5-1.5B-Instruct"
DEFAULT_COMP_CONFIG="configs/LightThinker/qwen/v1.json"
DEFAULT_MODEL_TYPE="qwen"
DEFAULT_DATASET="gsm8k"
DEFAULT_BOS_TOKEN="<|im_start|>"
DEFAULT_EOS_TOKEN="<|im_end|>"
DEFAULT_CACHE_SIZE=1024
DEFAULT_BASE_PATH=""
DEFAULT_INTERACTION="false"

# 帮助信息
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --method            Method type (default: ${DEFAULT_METHOD})
                        Options: 'anchor-token', 'normal', 'kvcache', 'anchor-thought'
    --tokenizer_path    Path to tokenizer (default: ${DEFAULT_TOKENIZER_PATH})
    --comp_config       Compression config path (default: ${DEFAULT_COMP_CONFIG})
    --model_type        Model type (default: ${DEFAULT_MODEL_TYPE})
    --dataset           Dataset name (default: ${DEFAULT_DATASET})
                        Options: gsm8k, gpqa, mmlu, bbh
    --bos_token         BOS token (default: ${DEFAULT_BOS_TOKEN})
    --eos_token         EOS token (default: ${DEFAULT_EOS_TOKEN})
    --cache_size        Cache size (default: ${DEFAULT_CACHE_SIZE})
    --base_path         Base path containing .jsonl files (REQUIRED)
    --interaction       Enable interaction mode (flag, default: disabled)
    --help              Show this help message

Example:
    $0 --base_path inference_results/inf_lighthinker_epl_r1distillqwen1.5b/gsm8k/1310
    $0 --method normal --dataset gpqa --base_path results/my_model/gpqa/2000
    $0 --base_path results/exp1 --dataset mmlu --interaction
EOF
    exit 0
}

# 解析命令行参数
method="$DEFAULT_METHOD"
tokenizer_path="$DEFAULT_TOKENIZER_PATH"
comp_config="$DEFAULT_COMP_CONFIG"
model_type="$DEFAULT_MODEL_TYPE"
dataset="$DEFAULT_DATASET"
bos_token="$DEFAULT_BOS_TOKEN"
eos_token="$DEFAULT_EOS_TOKEN"
cache_size="$DEFAULT_CACHE_SIZE"
base_path="$DEFAULT_BASE_PATH"
interaction="$DEFAULT_INTERACTION"

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            method="$2"
            shift 2
            ;;
        --tokenizer_path)
            tokenizer_path="$2"
            shift 2
            ;;
        --comp_config)
            comp_config="$2"
            shift 2
            ;;
        --model_type)
            model_type="$2"
            shift 2
            ;;
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --bos_token)
            bos_token="$2"
            shift 2
            ;;
        --eos_token)
            eos_token="$2"
            shift 2
            ;;
        --cache_size)
            cache_size="$2"
            shift 2
            ;;
        --base_path)
            base_path="$2"
            shift 2
            ;;
        --interaction)
            interaction="true"
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$base_path" ]; then
    echo "Error: --base_path is required"
    echo "Use --help for usage information"
    exit 1
fi

# 检查路径是否存在
if [ ! -d "$base_path" ]; then
    echo "Error: Directory does not exist: $base_path"
    exit 1
fi

# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=========================================="
echo "Evaluation Configuration:"
echo "=========================================="
echo "Method:         $method"
echo "Model Type:     $model_type"
echo "Dataset:        $dataset"
echo "Base Path:      $base_path"
echo "Cache Size:     $cache_size"
echo "Interaction:    $interaction"
echo "=========================================="
echo ""

# 自动查找所有 .jsonl 文件
echo "Searching for .jsonl files in: $base_path"
files=()

# 使用 find 查找所有 .jsonl 文件并排序
while IFS= read -r -d '' file; do
    files+=("$file")
done < <(find "$base_path" -maxdepth 1 -name "*.jsonl" -type f -print0 | sort -z)

# 检查是否找到文件
if [ ${#files[@]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✗ Error: No .jsonl files found in $base_path"
    echo "=========================================="
    exit 1
fi

echo "Found ${#files[@]} .jsonl file(s):"
for file in "${files[@]}"; do
    echo "  ✓ $(basename "$file")"
done

echo ""
echo "=========================================="
echo "Starting evaluation with ${#files[@]} file(s)..."
echo "=========================================="

# 构建 Python 命令参数数组
python_args=(
    "evaluation/eval_file.py"
    "--method" "$method"
    "--tokenizer_path" "$tokenizer_path"
    "--comp_config" "$comp_config"
    "--model_type" "$model_type"
    "--dataset" "$dataset"
    "--files" "${files[@]}"
    "--cache_size" "$cache_size"
    "--bos_token" "$bos_token"
    "--eos_token" "$eos_token"
)

# 如果启用 interaction，添加该参数
if [ "$interaction" = "true" ]; then
    python_args+=("--interaction")
fi

# # 打印命令（用于调试）
# echo "Command:"
# echo "python \\"
# echo "  evaluation/eval_file.py \\"
# echo "  --method $method \\"
# echo "  --dataset $dataset \\"
# echo "  --files \\"
# for file in "${files[@]}"; do
#     echo "    $file \\"
# done | sed '$ s/ \\$//'
# echo ""

# 执行命令
python "${python_args[@]}"

# 检查执行结果
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Evaluation completed successfully!"
    echo "  Processed ${#files[@]} file(s) from:"
    echo "  $base_path"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Evaluation failed with error code: $exit_code"
    echo "=========================================="
    exit $exit_code
fi


# bash lxy_evaluate.sh --method anchor-thought --dataset bbh --base_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/infer_data_case_study/inf_lightthinker_r1distillqwen1.5b/bbh/1305
# bash lxy_evaluate.sh --method normal --dataset bbh --base_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/RRcot/sglang_inference_results/inf_baseline_r1distillqwen1.5b/bbh