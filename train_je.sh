
# ==================== 训练参数（可直接在脚本内修改） ====================
# 直接运行: bash train_je.sh
# 可选覆盖: bash train_je.sh [root_dir] [init_tag] [use_EPL] [lr] [mode] [aux_config] [output_base_dir] [tokenizer_path] [model_path] [train_data_path]

# 默认参数（推荐在这里改）
root_dir="/mnt/jinbo/RLRM/RRcot"
init_tag="epl_apa_true"
use_EPL="True" #vanilla和lightthinker为false，其余为true
lr="2e-5" #vanilla 1e-5  anllm lightthinker 2e-5
mode="aug-wo-pc" #aug-wo-pc, normal
aux_config="None"
output_base_dir="/mnt/jinbo/RLRM/RRcot/output"
tokenizer_path="/mnt/jinbo/RLRM/model/Qwen/Qwen2.5-7B-Instruct"
model_path="/mnt/jinbo/RLRM/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
train_data_path="/mnt/jinbo/RLRM/RRcot/data/train/train.jsonl"

# deepspeed include 策略: auto(根据 CUDA_VISIBLE_DEVICES 自动生成) 或 fixed
ds_include_mode="auto"
fixed_include="localhost:0,1,2,3"

# 可选: 使用命令行参数覆盖默认参数
if [ $# -gt 0 ]; then
    if [ $# -lt 9 ]; then
        echo "错误: 传参模式下缺少必需参数"
        echo "使用方法: $0 [root_dir] [init_tag] [use_EPL] [lr] [mode] [aux_config] [output_base_dir] [tokenizer_path] [model_path] [train_data_path]"
        exit 1
    fi

    root_dir="$1"
    init_tag="$2"
    use_EPL="$3"
    lr="$4"
    mode="$5"
    aux_config="${6:-None}"
    output_base_dir="$7"
    tokenizer_path="$8"
    model_path="$9"
    train_data_path="${10}"
fi

# 检查必需参数是否为空
if [ -z "$root_dir" ] || [ -z "$init_tag" ] || [ -z "$use_EPL" ] || [ -z "$lr" ] || [ -z "$mode" ] || [ -z "$output_base_dir" ] || [ -z "$tokenizer_path" ] || [ -z "$model_path" ] || [ -z "$train_data_path" ]; then
    echo "错误: root_dir, init_tag, use_EPL, lr, mode, output_base_dir, tokenizer_path, model_path, train_data_path 不能为空"
    echo "root_dir: $root_dir"
    echo "init_tag: $init_tag"
    echo "use_EPL: $use_EPL"
    echo "lr: $lr"
    echo "mode: $mode"
    echo "output_base_dir: $output_base_dir"
    echo "tokenizer_path: $tokenizer_path"
    echo "model_path: $model_path"
    echo "train_data_path: $train_data_path"
    exit 1
fi

# 处理 aux_config
if [ "$aux_config" = "None" ] || [ -z "$aux_config" ]; then
    aux_config="None"
else
    # 如果 aux_config 是相对路径，则基于 root_dir 转换为绝对路径
    if [[ ! "$aux_config" = /* ]]; then
        aux_config="$root_dir/$aux_config"
    fi
fi

# 检查 root_dir 是否存在
if [ ! -d "$root_dir" ]; then
    echo "错误: root_dir 不存在: $root_dir"
    exit 1
fi

cd $root_dir

# ========================= 保存模型路径 =============================
output_dir="${output_base_dir}/${init_tag}/train"
# 创建输出目录
mkdir -p "$output_dir"
# ========================= 保存模型路径 =============================

# ========================= 训练日志路径 =============================
# 生成带时间戳的日志文件名
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${output_dir}/train_${timestamp}.log"
# 同时创建一个最新的日志文件链接
latest_log="${output_dir}/train_latest.log"
# ========================= 训练日志路径 =============================

export PYTHONPATH=$PYTHONPATH:$(pwd)
# model 
model_type="qwen"
# tokenizer_path, model_path, train_data_path 从命令行参数传入
bos_token="<|im_start|>"
eos_token="<|im_end|>"
conf_version="adaptive_v1"

# training
max_length=4096
lr_scheduler_type="cosine"
epochs=5   #change to 1 for test
# lr 从命令行参数传入，不再硬编码
save_steps=2
deepspeed="$root_dir/configs/ds_z3_offload_config.json"
micro_batch_size=1
gradient_accumulation_steps=4
warmup_ratio=0.05
# mode 从命令行参数传入，不再硬编码
warmup_steps=0

# others
model_size="1.5b"
# train_path 从命令行参数传入
train_path="$train_data_path"
see_current="false"
bi_directional="false"
diagonal="false"
exclude_continue="false"
qkv="no"
freeze_model="false"
train_on_input="false"
hybrid="false"
output_compress_instruction="None"
prefill_compress="false"


echo "root_dir=${root_dir}"
echo "init_tag=${init_tag}"
echo "use_EPL=${use_EPL}"
echo "lr=${lr}"
echo "mode=${mode}"
echo "aux_config=${aux_config}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "model_type=${model_type}"
echo "model_size=${model_size}"
echo "model_path=${model_path}"
echo "tokenizer_path=${tokenizer_path}"
echo "train_path=${train_path}"
echo "lr_scheduler_type=${lr_scheduler_type}"
echo "max_length=${max_length}"
echo "bos_token=${bos_token}"
echo "eos_token=${eos_token}"
echo "see_current=${see_current}"
echo "bi_directional=${bi_directional}"
echo "diagonal=${diagonal}"
echo "mode=${mode}"
echo "exclude_continue=${exclude_continue}"
echo "qkv=${qkv}"
echo "freeze_model=${freeze_model}"
echo "train_on_input=${train_on_input}"
echo "hybrid=${hybrid}"
echo "output_compress_instruction=${output_compress_instruction}"
echo "prefill_compress=${prefill_compress}"
echo "epochs=${epochs}"
echo "lr=${lr}"
echo "save_steps=${save_steps}"
echo "deepspeed=${deepspeed}"
echo "micro_batch_size=${micro_batch_size}"
echo "gradient_accumulation_steps=${gradient_accumulation_steps}"
echo "warmup_ratio=${warmup_ratio}"
echo "warmup_steps=${warmup_steps}"

# att_info="${model_size}-${model_type}-len_${max_length}-see_cur_${see_current}-bi_${bi_directional}-diag_${diagonal}-mode_${mode}"
# train_info="prefill_compress_${prefill_compress}-hybrid_${hybrid}-epoch_${epochs}-lr_${lr}-bsz_${micro_batch_size}-accumu_${gradient_accumulation_steps}-warm_r_${warmup_ratio}-warm_s_${warmup_steps}-freeze_model_${freeze_model}-train_input_${train_on_input}-qkv_${qkv}-ex_con_${exclude_continue}"

compress_config="$root_dir/configs/LightThinker/${model_type}/${conf_version}.json"

if [ "$ds_include_mode" = "auto" ] && [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # auto 模式下，若已设置 CUDA_VISIBLE_DEVICES，则不再传 --include，避免 deepspeed 忽略可见卡
    use_include="false"
    echo "ds_include=disabled (use CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
else
    use_include="true"
    ds_include="$fixed_include"
    echo "ds_include=${ds_include}"
fi

# 使用 tee 命令同时输出到终端和日志文件
if [ "$use_include" = "true" ]; then
deepspeed --include "$ds_include" LightThinker/train.py \
    --model_type $model_type \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --train_path $train_path \
    --output_dir $output_dir \
    --max_length $max_length \
    --compress_config $compress_config \
    --bos_token $bos_token \
    --eos_token $eos_token \
    --see_current $see_current \
    --bi_directional $bi_directional \
    --diagonal $diagonal \
    --mode $mode \
    --exclude_continue $exclude_continue \
    --qkv $qkv \
    --freeze_model $freeze_model \
    --train_on_input $train_on_input \
    --output_compress_instruction $output_compress_instruction \
    --epochs $epochs \
    --lr $lr \
    --save_steps $save_steps \
    --deepspeed $deepspeed \
    --micro_batch_size $micro_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --warmup_ratio $warmup_ratio \
    --warmup_steps $warmup_steps \
    --hybrid $hybrid \
    --prefill_compress $prefill_compress \
    --lr_scheduler_type $lr_scheduler_type \
    --use_EPL $use_EPL \
    --aux_config $aux_config 2>&1 | tee "$log_file"
else
deepspeed LightThinker/train.py \
    --model_type $model_type \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --train_path $train_path \
    --output_dir $output_dir \
    --max_length $max_length \
    --compress_config $compress_config \
    --bos_token $bos_token \
    --eos_token $eos_token \
    --see_current $see_current \
    --bi_directional $bi_directional \
    --diagonal $diagonal \
    --mode $mode \
    --exclude_continue $exclude_continue \
    --qkv $qkv \
    --freeze_model $freeze_model \
    --train_on_input $train_on_input \
    --output_compress_instruction $output_compress_instruction \
    --epochs $epochs \
    --lr $lr \
    --save_steps $save_steps \
    --deepspeed $deepspeed \
    --micro_batch_size $micro_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --warmup_ratio $warmup_ratio \
    --warmup_steps $warmup_steps \
    --hybrid $hybrid \
    --prefill_compress $prefill_compress \
    --lr_scheduler_type $lr_scheduler_type \
    --use_EPL $use_EPL \
    --aux_config $aux_config 2>&1 | tee "$log_file"
fi

# 训练完成后，创建最新日志文件的软链接
train_exit_code=${PIPESTATUS[0]}  # 获取 deepspeed 命令的退出码
if [ -f "$log_file" ]; then
    # 使用绝对路径创建软链接
    cd "$output_dir"
    ln -sf "$(basename "$log_file")" "$(basename "$latest_log")"
    cd "$root_dir"
fi

# 返回训练命令的退出码
exit $train_exit_code
