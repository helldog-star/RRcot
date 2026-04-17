# ==================== 评估参数（可直接在脚本内修改） ====================
# 直接运行: bash evaluate_je.sh
# 可选覆盖: bash evaluate_je.sh [method] [tokenizer_path] [dataset] [base_path] [root_dir] [comp_config] [model_type] [bos_token] [eos_token] [cache_size] [interaction]

# 默认参数（推荐在这里改）
method="anchor-thought" # 评估方法，例如 "anchor-thought"、"normal" 等
tokenizer_path="/mnt/jinbo/RLRM/model/Qwen/Qwen2.5-7B-Instruct"
dataset="bbh"  # 数据集名称，例如 "mmlu"、"gsm8k"、"gpqa"、"bbh"
base_path="/mnt/jinbo/RLRM/RRcot/output/epl_apa_true/inference/${dataset}"
root_dir="/mnt/jinbo/RLRM/RRcot"
comp_config="configs/LightThinker/qwen/adaptive_v1.json"
model_type="qwen"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
cache_size="1024"
interaction="false"

# 可选: 使用命令行参数覆盖默认参数
if [ $# -gt 0 ]; then
    if [ $# -lt 4 ]; then
        echo "错误: 传参模式下缺少必需参数"
        echo "使用方法: $0 [method] [tokenizer_path] [dataset] [base_path] [root_dir] [comp_config] [model_type] [bos_token] [eos_token] [cache_size] [interaction]"
        exit 1
    fi

    method="$1"
    tokenizer_path="$2"
    dataset="$3"
    base_path="$4"
    root_dir="${5:-$root_dir}"
    comp_config="${6:-$comp_config}"
    model_type="${7:-$model_type}"
    bos_token="${8:-$bos_token}"
    eos_token="${9:-$eos_token}"
    cache_size="${10:-$cache_size}"
    interaction="${11:-$interaction}"
fi

# 检查必需参数是否为空
if [ -z "$method" ] || [ -z "$tokenizer_path" ] || [ -z "$dataset" ] || [ -z "$base_path" ] || [ -z "$root_dir" ]; then
    echo "错误: method, tokenizer_path, dataset, base_path, root_dir 不能为空"
    echo "method: $method"
    echo "tokenizer_path: $tokenizer_path"
    echo "dataset: $dataset"
    echo "base_path: $base_path"
    echo "root_dir: $root_dir"
    exit 1
fi

# 检查 root_dir 是否存在
if [ ! -d "$root_dir" ]; then
    echo "错误: root_dir 不存在: $root_dir"
    exit 1
fi

# 检查路径是否存在
if [ ! -d "$base_path" ]; then
    echo "错误: 目录不存在: $base_path"
    exit 1
fi

# 如果 comp_config 是相对路径，基于 root_dir 转成绝对路径
if [[ ! "$comp_config" = /* ]]; then
    comp_config="$root_dir/$comp_config"
fi

cd "$root_dir"

# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$root_dir

# 从 base_path 中提取模型标识（model_tag）
# base_path 格式通常是: {output_base_dir}/{model_tag}/inference/{dataset}
# 或者: {output_base_dir}/{model_tag}/{dataset}
model_tag=""
if [[ "$base_path" =~ /inference/ ]]; then
    # 如果包含 /inference/，提取 inference 之前的最后一个路径组件
    model_tag=$(echo "$base_path" | sed 's|.*/\([^/]*\)/inference/.*|\1|')
elif [[ "$base_path" =~ /rrcot/ ]]; then
    # 如果包含 /rrcot/，提取 rrcot 之后的第一个路径组件
    model_tag=$(echo "$base_path" | sed 's|.*/rrcot/\([^/]*\)/.*|\1|')
else
    # 如果无法提取，使用 base_path 的父目录名作为模型标识
    model_tag=$(basename "$(dirname "$base_path")")
fi

# 如果 model_tag 为空或等于 dataset，尝试其他方式提取
if [ -z "$model_tag" ] || [ "$model_tag" = "$dataset" ]; then
    # 尝试从路径中提取倒数第二个组件
    path_parts=($(echo "$base_path" | tr '/' ' '))
    if [ ${#path_parts[@]} -ge 2 ]; then
        model_tag="${path_parts[-2]}"
    else
        model_tag="unknown"
    fi
fi

# 构建评估结果保存路径（模型路径在前，数据集路径在后）
eval_result_dir="eval_results/${method}/${model_type}/${model_tag}/${dataset}"
mkdir -p "$eval_result_dir"
timestamp=$(date +"%Y%m%d_%H%M%S")
eval_log_file="${eval_result_dir}/eval_${timestamp}.log"
eval_latest_log="${eval_result_dir}/eval_latest.log"

# 定义日志记录函数
log_and_print() {
    echo "$@" | tee -a "$eval_log_file"
}

# 开始记录日志
log_and_print "=========================================="
log_and_print "评估配置:"
log_and_print "=========================================="
log_and_print "method: ${method}"
log_and_print "model_tag: ${model_tag}"
log_and_print "tokenizer_path: ${tokenizer_path}"
log_and_print "dataset: ${dataset}"
log_and_print "base_path: ${base_path}"
log_and_print "comp_config: ${comp_config}"
log_and_print "model_type: ${model_type}"
log_and_print "bos_token: ${bos_token}"
log_and_print "eos_token: ${eos_token}"
log_and_print "cache_size: ${cache_size}"
log_and_print "interaction: ${interaction}"
log_and_print "评估日志: ${eval_log_file}"
log_and_print "=========================================="
log_and_print ""

# 自动查找所有 .jsonl 文件
log_and_print "在目录中搜索 .jsonl 文件: $base_path"
files=()

# 使用 find 查找所有 .jsonl 文件并排序
while IFS= read -r -d '' file; do
    files+=("$file")
done < <(find "$base_path" -maxdepth 1 -name "*.jsonl" -type f -print0 | sort -z)

# 检查是否找到文件
if [ ${#files[@]} -eq 0 ]; then
    log_and_print ""
    log_and_print "=========================================="
    log_and_print "✗ 错误: 在 $base_path 中未找到 .jsonl 文件"
    log_and_print "=========================================="
    exit 1
fi

log_and_print "找到 ${#files[@]} 个 .jsonl 文件:"
for file in "${files[@]}"; do
    log_and_print "  ✓ $(basename "$file")"
done

log_and_print ""
log_and_print "=========================================="
log_and_print "开始评估，共 ${#files[@]} 个文件..."
log_and_print "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_and_print "=========================================="

# 构建 Python 命令参数数组
python_args=(
    "evaluation/eval_file.py"
    "--method" "$method"
    "--tokenizer_path" "$tokenizer_path"
    "--comp_config" "$comp_config"
    "--model_type" "$model_type"
    "--dataset" "$dataset"
    "--model_tag" "$model_tag"
    "--files" "${files[@]}"
    "--cache_size" "$cache_size"
    "--bos_token" "$bos_token"
    "--eos_token" "$eos_token"
)

# 如果启用 interaction，添加该参数
if [ "$interaction" = "true" ]; then
    python_args+=("--interaction")
fi

# 执行命令，使用 tee 同时输出到控制台和日志文件
python "${python_args[@]}" 2>&1 | tee -a "$eval_log_file"

# 检查执行结果
eval_exit_code=${PIPESTATUS[0]}  # 获取 python 命令的退出码
log_and_print ""
log_and_print "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_and_print "=========================================="

if [ $eval_exit_code -eq 0 ]; then
    # 创建最新日志文件的软链接
    if [ -f "$eval_log_file" ]; then
        cd "$eval_result_dir"
        ln -sf "$(basename "$eval_log_file")" "$(basename "$eval_latest_log")"
        cd - > /dev/null
    fi
    
    log_and_print ""
    log_and_print "=========================================="
    log_and_print "✓ 评估完成成功！"
    log_and_print "  处理了 ${#files[@]} 个文件，来自:"
    log_and_print "  $base_path"
    log_and_print ""
    log_and_print "评估结果和日志已保存到:"
    log_and_print "  评估结果: ${eval_result_dir}/result.txt"
    log_and_print "  评估日志: ${eval_log_file}"
    log_and_print "  最新日志: ${eval_latest_log} (软链接)"
    if [ -f "${eval_result_dir}/frequency.jsonl" ]; then
        log_and_print "  频率统计: ${eval_result_dir}/frequency.jsonl"
    fi
    log_and_print "=========================================="
else
    log_and_print ""
    log_and_print "=========================================="
    log_and_print "✗ 评估失败，错误代码: $eval_exit_code"
    log_and_print "  日志已保存到: $eval_log_file"
    log_and_print "=========================================="
    exit $eval_exit_code
fi
