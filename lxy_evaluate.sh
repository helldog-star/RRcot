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
DEFAULT_FOLDER="inf_lightthinker_r1distillqwen1.5b"
DEFAULT_CKPT=1305
DEFAULT_NUM_FILES=4
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
    --folder            Result folder name (default: ${DEFAULT_FOLDER})
    --ckpt              Checkpoint number (default: ${DEFAULT_CKPT})
    --num_files         Number of files to combine (default: ${DEFAULT_NUM_FILES})
    --interaction       Enable interaction mode (flag, default: disabled)
    --help              Show this help message

Example:
    $0 --dataset mmlu --ckpt 2000 --num_files 8
    $0 --method normal --dataset gpqa --num_files 2 --interaction
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
folder="$DEFAULT_FOLDER"
ckpt="$DEFAULT_CKPT"
num_files="$DEFAULT_NUM_FILES"
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
        --folder)
            folder="$2"
            shift 2
            ;;
        --ckpt)
            ckpt="$2"
            shift 2
            ;;
        --num_files)
            num_files="$2"
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

# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 构建文件路径数组
files=()
base_path="inference_results/${folder}/${dataset}/${ckpt}"

echo "=========================================="
echo "Evaluation Configuration:"
echo "=========================================="
echo "Method:         $method"
echo "Model Type:     $model_type"
echo "Dataset:        $dataset"
echo "Checkpoint:     $ckpt"
echo "Folder:         $folder"
echo "Num Files:      $num_files"
echo "Cache Size:     $cache_size"
echo "Interaction:    $interaction"
echo "=========================================="
echo ""

# 自动生成文件路径
echo "Generating file paths..."
for ((i=1; i<=num_files; i++)); do
    file_path="${base_path}/${i}-${num_files}${folder}.jsonl"
    
    # 只添加存在的文件
    if [ -f "$file_path" ]; then
        files+=("$file_path")
        echo "  ✓ File $i: $file_path"
    else
        echo "  ✗ File $i: $file_path (NOT FOUND - SKIPPED)"
    fi
done

echo ""
echo "=========================================="
echo "Starting evaluation..."
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

# 执行命令
echo "Executing command:"
echo "python ${python_args[@]}"
echo ""

python "${python_args[@]}"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Evaluation completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Evaluation failed with error code: $?"
    echo "=========================================="
    exit 1
fi

# bash lxy_evaluate.sh --method anchor-thought --dataset gsm8k --folder inf_lighthinker_epl_r1distillqwen1.5b --ckpt 1310 --num_files 36
# bash lxy_evaluate.sh --method anchor-thought --dataset mmlu --folder inf_lightthinker_r1distillqwen1.5b --ckpt 1305 --num_files 4
# bash lxy_evaluate.sh --method anchor-thought --dataset mmlu --folder inf_lightthinker_r1distillqwen7b --ckpt 1305 --num_files 16

# # 评估sglang推理的norml
# method="normal"
# dataset="mmlu" # gsm8k gpqa mmlu bbh
# # folder="inf_baseline_r1distillqwen1.5b_fix"
# folder="inf_baseline_r1distillqwen7b"
# file="sglang_inference_results/${folder}/${dataset}_result.jsonl"
# python evaluation/eval_file.py \
#   --method $method \
#   --tokenizer_path $tokenizer_path \
#   --comp_config $comp_config \
#   --model_type $model_type \
#   --dataset $dataset \
#   --files $file \
#   --cache_size $cache_size \
#   --bos_token $bos_token \
#   --eos_token $eos_token 
#   # --interaction 