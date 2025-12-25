import os
import json
import time
import asyncio  # 添加 asyncio 导入
import argparse
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer  

import sglang as sgl
from dataset_reader import MMLUReader, BBHReader, GSM8KReader, GPQAReader
from dataset_reader_cot import MMLUCOTReader, BBHCOTReader, GSM8KCOTReader, GPQACOTReader

DATASET_MAPPING = {
    "mmlu": MMLUReader,
    "bbh": BBHReader,
    "gsm8k": GSM8KReader,
    "gpqa": GPQAReader,
    "mmlu_cot": MMLUCOTReader,
    "bbh_cot": BBHCOTReader,
    "gsm8k_cot": GSM8KCOTReader,
    "gpqa_cot": GPQACOTReader
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen2.5 model")
    parser.add_argument("--datasets", nargs="+", default=["mmlu", "gsm8k", "gpqa", "bbh"], help="Datasets to run")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor Parallelism size")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for progress bar updates")
    parser.add_argument("--extend_name", type=str, default="Your_tag", help="Sub-folder name")
    return parser.parse_args()

async def main_async():  # 改为 async 函数
    args = get_args()
    
    # 2. 初始化 Tokenizer
    print(f"[Init] Loading Tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 3. 初始化 SGLang 引擎
    print(f"[Init] Loading Engine from {args.model_path} with TP={args.tp_size}...")
    engine = sgl.Engine(
        model_path=args.model_path,
        tp_size=args.tp_size,
        trust_remote_code=True,
        dtype="bfloat16", 
    )

    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": 10240,
        "top_p": 1.0,
        "top_k": -1,
        "skip_special_tokens": False,
    }

    final_save_dir = os.path.join(args.output_dir, args.extend_name)
    os.makedirs(final_save_dir, exist_ok=True)

    try:  # 添加 try-finally 确保资源释放
        for dataset_name in args.datasets:
            if dataset_name not in DATASET_MAPPING:
                print(f"[Warning] Dataset {dataset_name} not found, skipping.")
                continue
            
            print(f"\n[Run] Preparing dataset: {dataset_name}")
            
            ReaderClass = DATASET_MAPPING[dataset_name]
            reader = ReaderClass()
            dataset_len = len(reader)
            
            prompts_text_batch = []
            original_messages_batch = []
            indices = []
            gt_answers = []

            system_prompt = reader.get_system_prompt()

            # 4. 数据预处理
            print(f"      Applying chat template for {dataset_len} samples...")
            for i in range(dataset_len):
                user_prompt = reader.get_prompt(i)
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
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
            
            for i in tqdm(range(0, len(prompts_text_batch), args.batch_size), desc=f"Infer {dataset_name}"):
                batch_prompts = prompts_text_batch[i : i + args.batch_size]
                
                # 批量推理：传入 list
                batch_outputs = await engine.async_generate(
                    batch_prompts,
                    sampling_params=sampling_params
                )
                
                # 处理输出：如果返回单个 dict，转为 list
                if isinstance(batch_outputs, dict):
                    batch_outputs = [batch_outputs]
                elif not isinstance(batch_outputs, list):
                    # 如果是其他格式，尝试转换
                    batch_outputs = list(batch_outputs)
                
                all_outputs.extend(batch_outputs)
            
            total_time = time.time() - start_time
            avg_time = total_time / len(prompts_text_batch) if len(prompts_text_batch) > 0 else 0
            print(f"      Inference finished. Total time: {total_time:.2f}s, Avg: {avg_time:.4f}s/sample")

            # 6. 保存结果
            # 这里不同评测集分文件夹保存，与lightthinker一致，方便统一评估
            dataset_output_dir = os.path.join(final_save_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
        
            out_file = os.path.join(dataset_output_dir, f"{dataset_name}_result.jsonl")
            # out_file = os.path.join(final_save_dir, f"{dataset_name}_result.jsonl")
            print(f"      Processing results and saving to {out_file}...")
            
            with open(out_file, "w", encoding="utf-8") as f:
                for i, output_obj in enumerate(all_outputs):
                    idx = indices[i]
                    gt = gt_answers[i]
                    messages = original_messages_batch[i]
                    prompt_text = prompts_text_batch[i]

                    # 提取生成文本
                    model_output = output_obj.get("text", "")
                    
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
                        print(f"[Warning] Error evaluating idx {idx}: {e}")
                        is_correct = False

                    extracted_answer = reader.extract_answer(model_output)

                    result_entry = {
                        "idx": idx,
                        "prompt": prompt_text,
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
                    
                    f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

            print(f"      Done. Results saved to {out_file}")
    
    finally:
        # 7. 关闭引擎
        print("\n[Cleanup] Shutting down engine...")
        engine.shutdown()

def main():
    """同步包装函数"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()