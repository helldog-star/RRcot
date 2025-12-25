export PYTHONPATH=$PYTHONPATH:$(pwd)
# we will load model from `output/{model_tag}/checkpoint-{args.ckpt}`

# `model_tag` is the filename under the output/ folder, 
# corresponding to line 1461 of the code in LightThinker/inference.py.
# model_tag="lighthinker_7b_aug-wo-pc"
model_tag="lighthinker-epl_1d5b_aug-wo-pc"

# `model_short_tag` is used to save file, 
# corresponding to line 1691 of the code in LightThinker/inference.py.
model_short_tag="inf_lightthinker_epl_r1distillqwen1.5b"

model_type="qwen"
tokenizer_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen2.5-1.5B-Instruct"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
compress_config="./configs/LightThinker/qwen/v1.json"

ckpt=1310
# output_tag="inf_lightthinker_r1distillqwen1.5b"
output_tag="inf_lightthinker_epl_r1distillqwen1.5b"
# `model_path` is an optional argument
# if you set the `model_path`, the arguments `ckpt` and `model_tag` will be ignored.
# see line 1460 of the code in LightThinker/inference.py for more details.
# model_path="/mnt/jinbo/RLRM/lightthinker/output/cosine1.5b-qwen-len_4096-see_cur_false-bi_false-diag_false-mode_aug-wo-pc-prefill_compress_false-hybrid_false-epoch_5-lr_2e-5-bsz_1-accumu_4-warm_r_0.05-warm_s_0-freeze_model_false-train_input_false-qkv_no-ex_con_false/checkpoint-5220"
max_new_tokens=10240

root_dir="./LightThinker"

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

# check "ours_infer_log" 
if [ ! -d "ours_infer_log" ]; then
    echo "Creating ours_infer_log directory..."
    mkdir ours_infer_log
fi
subfolders=("true_true" "true_false" "false_false" "false_true")
for subfolder in "${subfolders[@]}"; do
    if [ ! -d "ours_infer_log/$subfolder" ]; then
        echo "Creating $subfolder directory..."
        mkdir "ours_infer_log/$subfolder"
    fi
done

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

        CUDA_VISIBLE_DEVICES=$device nohup python "${root_dir}/inference.py" \
            --model_tag $model_tag \
            --model_short_tag $model_short_tag \
            --ckpt $ckpt \
            --tokenizer_path $tokenizer_path \
            --compress_config $compress_config \
            --max_new_tokens $max_new_tokens \
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
            --index $real_index > "ours_infer_log/${rolling_rope}_${compress_prompt}/${real_index}${prefix}_${model_short_tag}_${ckpt}.txt" 2>&1 &
        
        sleep 5
    done
    ((logical_id++))
done
