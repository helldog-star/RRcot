import os
import json
import time
import argparse
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer  # 新增：用于手动应用 chat template

# 1. 设定 GPU 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sglang as sgl
# 引用 dataset_reader.py
from dataset_reader import MMLUReader, BBHReader, GSM8KReader, GPQAReader

DATASET_MAPPING = {
    "mmlu": MMLUReader,
    "bbh": BBHReader,
    "gsm8k": GSM8KReader,
    "gpqa": GPQAReader,
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen2.5 model")
    parser.add_argument("--datasets", nargs="+", default=["mmlu", "gsm8k", "gpqa", "bbh"], help="Datasets to run")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor Parallelism size")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for progress bar updates")
    parser.add_argument("--extend_name", type=str, default="Your_tag", help="Set for ")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 2. 初始化 Tokenizer (用于处理 Prompt 格式)
    print(f"[Init] Loading Tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 3. 初始化 SGLang 引擎
    print(f"[Init] Loading Engine from {args.model_path} with TP={args.tp_size}...")
    engine = sgl.Engine(
        model_path=args.model_path,
        tp_size=args.tp_size,
        trust_remote_code=True,
        dtype="bfloat16", # Qwen2.5 推荐
    )

    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 10240,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_MAPPING:
            print(f"[Warning] Dataset {dataset_name} not found, skipping.")
            continue
        
        print(f"\n[Run] Preparing dataset: {dataset_name}")
        
        ReaderClass = DATASET_MAPPING[dataset_name]
        reader = ReaderClass()
        dataset_len = len(reader)
        
        # 准备数据容器
        prompts_text_batch = [] # 存放处理好模板的字符串
        original_messages_batch = [] # 存放原始 messages 结构用于保存
        indices = []
        gt_answers = []

        system_prompt = reader.get_system_prompt()

        # 4. 数据预处理：提前将 Chat 格式转为 String
        print(f"      Applying chat template for {dataset_len} samples...")
        for i in range(dataset_len):
            user_prompt = reader.get_prompt(i)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            

            # 关键修复：手动应用 Chat Template
            # tokenize=False 表示只生成字符串，不转 id，因为 sglang engine 接收字符串
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            if i == 0:
                print("\n[DEBUG] Final Prompt sent to model (First Sample):")
                print("-" * 50)
                print(full_prompt)
                print("-" * 50)

            prompts_text_batch.append(full_prompt)
            original_messages_batch.append(messages)
            indices.append(i)
            gt_answers.append(reader.get_answer(i))

        print(f"      Starting inference...")

        # 5. 分批执行 generate
        all_outputs = []
        start_time = time.time()
        
        # tqdm 进度条
        for i in tqdm(range(0, len(prompts_text_batch), args.batch_size), desc=f"Infer {dataset_name}"):
            batch_prompts = prompts_text_batch[i : i + args.batch_size]
            
            # 关键修复：使用 generate 而不是 chat
            batch_outputs = engine.generate(
                batch_prompts,
                sampling_params=sampling_params
            )
            all_outputs.extend(batch_outputs)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(prompts_text_batch) if len(prompts_text_batch) > 0 else 0
        print(f"      Inference finished. Avg time per sample: {avg_time:.4f}s")

        out_file = os.path.join(args.output_dir+"/"+args.extend_name, f"{dataset_name}_result.jsonl")
        print(f"      Processing results and saving to {out_file}...")
        
        with open(out_file, "w", encoding="utf-8") as f:
            for i, output_obj in enumerate(all_outputs):
                idx = indices[i]
                gt = gt_answers[i]
                messages = original_messages_batch[i]
                
                model_output = output_obj["text"]
                
                # 获取 metadata
                meta_info = output_obj.get("meta_info", {})
                prompt_tokens = meta_info.get("prompt_tokens", 0)
                completion_tokens = meta_info.get("completion_tokens", 0)
                
                # 评测
                try:
                    acc_res = reader.get_acc(model_output, idx)
                    if isinstance(acc_res, tuple):
                        is_correct = acc_res[0]
                    else:
                        is_correct = acc_res
                except Exception as e:
                    is_correct = False

                extracted_answer = reader.extract_answer(model_output)

                result_entry = {
                    "idx": idx,
                    "model_answer": extracted_answer,
                    "gt_answer": gt,
                    "acc_state": is_correct,
                    "output": model_output,
                    "prompt_structure": messages,
                    "input_len": prompt_tokens,
                    "output_len": completion_tokens,
                    "infer_time": avg_time, 
                    "dataset": dataset_name
                }
                
                # 修改点：逐行写入 JSON 字符串
                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

        print(f"      Done.")

if __name__ == "__main__":
    main()