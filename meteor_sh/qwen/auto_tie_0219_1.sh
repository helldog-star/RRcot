#!/bin/bash

# ==================== 路径配置 ====================
# 所有路径统一在此设置，便于在不同服务器上运行
# ROOT_DIR="/zhaorunsong/RRcot"  # 项目根目录
ROOT_DIR="/mnt/zhaorunsong/lx/RRcot" 
INFERENCE_ROOT_DIR="${ROOT_DIR}/LightThinker"  # 推理脚本使用的代码根目录

# 输出路径配置
OUTPUT_BASE_DIR="/mnt/zhaorunsong/lx/rrcot_test"  # 所有输出（训练、推理）的基础目录

# 模型和Tokenizer路径配置
TOKENIZER_PATH="/mnt/zhaorunsong/models/Qwen2.5-0.5B-Instruct"  # Tokenizer路径
MODEL_PATH="/mnt/zhaorunsong/models/Qwen2.5-0.5B-Instruct"  # 预训练模型路径

# 训练数据路径配置
TRAIN_DATA_PATH="/mnt/zhaorunsong/lx/RRcot/data/train_test.jsonl"  # 训练数据路径

# Conda环境配置（用于sglang_inference.sh）
CONDA_SH_PATH="/mnt/zhaorunsong/anaconda3/etc/profile.d/conda.sh"  # Conda初始化脚本路径
# CONDA_SH_PATH="/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV_NAME="sglang"  # Conda环境名称

# ==================== 推理和评估配置 ====================
# 设置推理和评估的默认参数
REPETITION_PENALTY="1.1"  # 重复惩罚系数
CKPT="39"  # 检查点编号，可以根据实际情况修改
DATASETS=("bbh" "gpqa" "gsm8k" "mmlu")  # 要评估的数据集

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.sh"
INFERENCE_SCRIPT="${SCRIPT_DIR}/inference.sh"
SGLANG_INFERENCE_SCRIPT="${SCRIPT_DIR}/sglang_inference.sh"
EVALUATE_SCRIPT="${SCRIPT_DIR}/evaluate.sh"

# ==================== 通用函数 ====================

export PYTHONPATH=$PYTHONPATH:${ROOT_DIR}
cd ${ROOT_DIR}

# 训练模型
train_model() {
    local model_tag=$1
    local use_EPL=$2
    local lr=$3
    local mode=$4
    local aux_config=$5
    local conf_version=$6
    
    echo "=======🚀 ${model_tag}开始训练 ======="
    bash ${TRAIN_SCRIPT} "${ROOT_DIR}" "${model_tag}" "${use_EPL}" "${lr}" "${mode}" "${aux_config}" "${OUTPUT_BASE_DIR}" "${TOKENIZER_PATH}" "${MODEL_PATH}" "${TRAIN_DATA_PATH}" "${conf_version}"
    if [ $? -ne 0 ]; then
        echo "❌ ${model_tag}训练失败"
        return 1
    fi
    echo "=======✅ ${model_tag}结束训练 ======="
    return 0
}

# 推理和评估模型
inference_and_evaluate() {
    local model_tag=$1
    local eval_method=$2
    local inference_script_type=$3
    local compress_config=$4
    
    echo ""
    echo "=========================================="
    echo "      🚀 ${model_tag} 开始推理     "
    echo "=========================================="
    
    # 根据模型类型选择推理脚本
    if [ "$inference_script_type" = "sglang_inference" ]; then
        INFERENCE_CMD="${SGLANG_INFERENCE_SCRIPT}"
        echo "使用 sglang_inference.sh 进行推理"
        bash ${INFERENCE_CMD} "${model_tag}" "${REPETITION_PENALTY}" "${CKPT}" "${INFERENCE_ROOT_DIR}" "${OUTPUT_BASE_DIR}" "${CONDA_SH_PATH}" "${CONDA_ENV_NAME}"
    else
        INFERENCE_CMD="${INFERENCE_SCRIPT}"
        echo "使用 inference.sh 进行推理"
        bash ${INFERENCE_CMD} "${model_tag}" "${REPETITION_PENALTY}" "${CKPT}" "${INFERENCE_ROOT_DIR}" "${OUTPUT_BASE_DIR}" "${TOKENIZER_PATH}" "${compress_config}"
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ ${model_tag} 推理失败"
        return 1
    fi
    
    echo "=======✅ ${model_tag} 推理完成 ======="
    
    # 等待推理完全完成
    sleep 10
    
    # 运行评估
    echo ""
    echo "=========================================="
    echo "      🚀 ${model_tag} 评估开始     "
    echo "=========================================="
    
    output_path="${OUTPUT_BASE_DIR}/${model_tag}"
    output_tag="${output_path}/inference"
    
    for dataset in "${DATASETS[@]}"; do
        base_path="${output_tag}/${dataset}"
        
        if [ ! -d "$base_path" ]; then
            echo "⚠️  警告: 推理结果路径不存在: ${base_path}"
            echo "跳过 ${dataset} 数据集的评估"
            continue
        fi
        
        echo "评估数据集: ${dataset}"
        bash ${EVALUATE_SCRIPT} "${eval_method}" "${TOKENIZER_PATH}" "${dataset}" "${base_path}" "${comp_config}"
        
        if [ $? -ne 0 ]; then
            echo "❌ ${model_tag} 在 ${dataset} 数据集上评估失败"
        else
            echo "✅ ${model_tag} 在 ${dataset} 数据集上评估完成"
        fi
    done
    
    echo "=========================================="
    echo "      ✅ ${model_tag} 评估完成     "
    echo "=========================================="
    return 0
}



==================== 模型: epl_apa_mtp_w3e-1 ====================
# train_model "epl_apa_mtp_w3e-1" "True" "2e-5" "aug-wo-pc-apa-mtp" "configs/epl_apa_mtp.json" "apa_mtp"
# if [ $? -ne 0 ]; then
#     echo "❌ epl_apa_mtp_w3e-1训练失败，退出"
#     exit 1
# fi
inference_and_evaluate "epl_apa_mtp_w3e-1" "anchor-thought" "inference" "./configs/LightThinker/qwen/apa_mtp.json"


echo ""
echo "=========================================="
echo "      ✅ 所有模型训练、推理和评估完成     "
echo "=========================================="
