eval "$(/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/miniconda/bin/conda shell.bash hook)"
which conda
conda activate lightinfer
which python

export PYTHONPATH="$(pwd):${PYTHONPATH}"

model_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/RRcot/output/cosine1d5b-qwen-len_4096-see_cur_false-bi_false-diag_false-mode_normal-prefill_compress_false-hybrid_false-epoch_5-lr_1e-5-bsz_8-accumu_2-warm_r_0d05-warm_s_0-freeze_model_false-train_input_false-qkv_no-ex_con_false/checkpoint-1305"
datasets="mmlu gsm8k gpqa bbh"
batch_size=16
output_dir="./sglang_inference_results"
extend_name="inf_baseline_r1distillqwen1.5b"

root_dir="./LightThinker"

python "${root_dir}/sglang_inference.py" \
  --model_path $model_path \
  --datasets $datasets \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --extend_name $extend_name
