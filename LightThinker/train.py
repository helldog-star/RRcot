import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
import argparse
import torch
from typing import *
from tqdm import tqdm
from copy import deepcopy
from model_qwen import Qwen2ForCausalLM
from model_llama import LlamaForCausalLM
from transformers import Trainer, TrainingArguments
import deepspeed
import torch.distributed as dist
from datetime import timedelta  # 引入时间库
from transformers import TrainerCallback
from transformers.integrations import TensorBoardCallback


# ===== 关键：在任何其他操作前绑定GPU设备 =====
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank) 
    print(f"Process {os.getpid()} set to device: cuda:{local_rank}")
else:
    local_rank = 0
    print("Running in single-GPU mode")
# =========================================



if not torch.distributed.is_initialized():
    deepspeed.init_distributed(
        dist_backend="nccl",
        timeout=timedelta(minutes=120)
    )

from config import Config
from LightThinker.utils import _print, IGNORE_LABEL_ID, str2bool
from tokenizer import Tokenizer
from dataset import MyDataset, MyDataCollator

from transformers import TrainerCallback

class SaveTokenizerCallback(TrainerCallback):
    """保存checkpoint同时保存tokenizer"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def on_save(self, args, state, control, **kwargs):
        """在保存checkpoint时调用"""
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}"
        )
        
        if os.path.exists(checkpoint_folder):
            self.tokenizer.save_pretrained(checkpoint_folder)




# ==========================================
# 1. 定义外部存储容器 (用于暂存 Hook 抓取的数据)
# ==========================================
class LossBuffer:
    def __init__(self):
        self.history = []
    
    def clear(self):
        self.history = []

# 实例化全局 Buffer
loss_buffer = LossBuffer()

# ==========================================
# 2. 定义 Hook 函数 (用于从模型 forward 中偷数据)
# ==========================================
def capture_loss_hook(module, input, output):
    """
    修复版 Hook：自动处理 Tensor 和 float 类型不一致的问题
    """
    # 1. 获取属性，如果不存在则默认为 0.0
    raw_lm = getattr(module, '_last_lm_loss', 0.0)
    raw_mtp = getattr(module, '_last_mtp_loss', 0.0)
    
    # 2. 定义一个辅助逻辑：确保输出统一为 Tensor
    # 这样 Callback 里的 .item() 才能正常工作
    def safe_detach(val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu() # 移到 CPU 以节省显存
        else:
            return torch.tensor(float(val)) # 如果是 float，这就包装成 Tensor
            
    # 3. 存入 buffer
    loss_buffer.history.append({
        "lm": safe_detach(raw_lm),
        "mtp": safe_detach(raw_mtp)
    })

# ==========================================
# 3. 定义 Callback (用于汇总计算并写入 Log)
# ==========================================
class MTPLossCallback(TrainerCallback):
    def __init__(self, buffer_instance):
        self.buffer = buffer_instance
        self.local_stats = {"lm_sum": 0.0, "mtp_sum": 0.0, "micro_count": 0}

    def on_step_end(self, args, state, control, **kwargs):
        # 从外部 buffer 读取数据
        history = self.buffer.history
        
        if not history:
            return

        # 统计当前 optimizer step 内所有 micro-batch 的损失总和与数量
        local_step_lm_sum = sum(item["lm"].item() for item in history)
        local_step_mtp_sum = sum(item["mtp"].item() for item in history)
        micro_count = len(history)
        
        self.local_stats["lm_sum"] += local_step_lm_sum
        self.local_stats["mtp_sum"] += local_step_mtp_sum
        self.local_stats["micro_count"] += micro_count
        
        # 【关键】清空 Buffer，防止内存溢出和数据混淆
        self.buffer.clear()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self.local_stats["micro_count"] > 0:
            local_lm_sum = self.local_stats["lm_sum"]
            local_mtp_sum = self.local_stats["mtp_sum"]
            local_micro_count = float(self.local_stats["micro_count"])
            
            # === DeepSpeed 多卡同步核心逻辑 ===
            if dist.is_initialized():
                # 同步 sum 与 count，做全局加权平均，避免卡间样本数不一致带来的偏差
                metrics = torch.tensor(
                    [local_lm_sum, local_mtp_sum, local_micro_count],
                    dtype=torch.float64,
                    device=args.device,
                )
                
                # 所有显卡的数据相加 (ReduceOp.SUM)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                
                global_micro_count = max(metrics[2].item(), 1.0)
                global_avg_lm = (metrics[0] / global_micro_count).item()
                global_avg_mtp = (metrics[1] / global_micro_count).item()
            else:
                global_avg_lm = local_lm_sum / max(local_micro_count, 1.0)
                global_avg_mtp = local_mtp_sum / max(local_micro_count, 1.0)

            model = kwargs.get("model", None)
            if model is not None and hasattr(model, "module"):
                model = model.module
            mtp_lambda = float(getattr(model, "mtp_lambda", 1.0)) if model is not None else 1.0
            global_avg_total = global_avg_lm + mtp_lambda * global_avg_mtp

            # 写入 Log (Trainer 会自动发送给 TensorBoard)
            # 覆盖 Trainer 默认 loss，使终端显示口径与 lm_loss/mtp_loss 一致
            logs['loss'] = round(global_avg_total, 4)
            logs['lm_loss'] = round(global_avg_lm, 4)
            logs['mtp_loss'] = round(global_avg_mtp, 4)
            
            # 重置计数器
            self.local_stats = {"lm_sum": 0.0, "mtp_sum": 0.0, "micro_count": 0}


def init_mtp_from_last_layer(model) -> None:
    if not hasattr(model, "mtp_modules") or getattr(model, "mtp_depth", 0) <= 0:
        return
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return

    last_layer = model.model.layers[-1]
    with torch.no_grad():
        def _format_key_list(keys, max_items: int = 5) -> str:
            if not keys:
                return "[]"
            shown = keys[:max_items]
            suffix = "" if len(keys) <= max_items else f"...(+{len(keys) - max_items})"
            return f"{shown}{suffix}"

        def _log_load(idx: int, name: str, result) -> None:
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
            _print(
                f"init mtp[{idx}] {name}: "
                f"missing={len(missing)} { _format_key_list(missing) } "
                f"unexpected={len(unexpected)} { _format_key_list(unexpected) }"
            )

        for idx, mtp_module in enumerate(model.mtp_modules):
            res = mtp_module.self_attn.load_state_dict(last_layer.self_attn.state_dict(), strict=False)
            _log_load(idx, "self_attn", res)
            res = mtp_module.cross_attn.load_state_dict(last_layer.self_attn.state_dict(), strict=False)
            _log_load(idx, "cross_attn(from self_attn)", res)
            res = mtp_module.mlp.load_state_dict(last_layer.mlp.state_dict(), strict=False)
            _log_load(idx, "mlp", res)


def freeze_except_mtp(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "mtp" in name:
            param.requires_grad = True

    # sanity check
    print("Trainable parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help="just used for deepspeed.")
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--train_path', type=str, help='training dataset path')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--max_length', type=int, default=768)
    parser.add_argument('--model_type', type=str, choices=['qwen', 'llama'])

    parser.add_argument('--compress_config', type=str)
    parser.add_argument('--bos_token', type=str)
    parser.add_argument('--eos_token', type=str)
    parser.add_argument('--see_current', type=str2bool)
    parser.add_argument('--bi_directional', type=str2bool)
    parser.add_argument('--diagonal', type=str2bool)
    parser.add_argument('--mode', type=str, choices=['recover', 'normal', 'aug', 'aug-wo-pc', 'aug-wo-pc-apa-mtp'])
    parser.add_argument('--exclude_continue', type=str2bool)
    parser.add_argument('--qkv', type=str)
    parser.add_argument('--freeze_model', type=str2bool)
    parser.add_argument('--train_on_input', type=str2bool)
    parser.add_argument('--output_compress_instruction', type=str)
    parser.add_argument('--hybrid', type=str2bool)  
    parser.add_argument('--prefill_compress', type=str2bool, default=True)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--save_steps', type=int)
    parser.add_argument('--deepspeed', type=str, help="file path")
    parser.add_argument('--micro_batch_size', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--warmup_ratio', type=float, default=0.)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')

    parser.add_argument('--use_EPL', type=str2bool, default=False)
    parser.add_argument('--aux_config', type=str, default=None)
    args = parser.parse_args()
    return args

def get_model_and_tokenizer(
    args,
    comp_config:Config
) -> Tuple[Union[Qwen2ForCausalLM, LlamaForCausalLM], Tokenizer]:

    special_token_list:List[str] = list()
    special_token_desp_dict = dict()
    tokenizer: Tokenizer = Tokenizer(
        tokenizer_path=args.tokenizer_path if args.tokenizer_path != None else args.model_path,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        special_token_list=None,
        add_prefix_space=False,
    )
    assert len(comp_config.special_token_desp_list) == len(comp_config.special_token_name_list)
    for desp, token in zip(comp_config.special_token_desp_list, comp_config.special_token_name_list):
        if tokenizer.convert_tokens_to_ids(token) == None:
            special_token_list.append(token)
            # special_token_desp_list.append(desp)
            special_token_desp_dict[token] = desp
    if len(special_token_list) > 0:
        tokenizer.add_special_token(special_token_list)
    
    if args.model_type == 'llama':
        model_class = LlamaForCausalLM
    elif args.model_type == 'qwen':
        model_class = Qwen2ForCausalLM
    else:
        assert False, "We only support llama and qwen model."

    if args.aux_config is not None and args.aux_config != "None":
        _print(f"use ce + mtp loss...")
        assert os.path.exists(args.aux_config)
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        with open(args.aux_config, "r", encoding='utf-8') as f:
            mtp_params = json.load(f)
        _print(f"auxiliary mtp config={mtp_params}")

        if comp_config.forzen_model_train_mtp:
            mtp_params["forzen_model_train_mtp"] = True

        model_config.update(mtp_params)
        model = model_class.from_pretrained(
            args.model_path, config=model_config, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        if getattr(model_config, "init", False):
            _print(f"initialize mtp from last layer...")
            init_mtp_from_last_layer(model)
        if comp_config.forzen_model_train_mtp:
            freeze_except_mtp(model)
    
    else:
        _print(f"use ce loss...")
        model = model_class.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16
        )
    
    # 1. 挂载钩子 (核心步骤)
    hook_handle = model.register_forward_hook(capture_loss_hook)

    model.add_qkv(
        q='q' in args.qkv,
        k='k' in args.qkv,
        v='v' in args.qkv,
    )

    if model.model.config.vocab_size != len(tokenizer):
        # Expand the token embedding and lm_head
        _print(f"before.embedding.shape={model.model.embed_tokens.weight.shape}")
        _print(f"before.lm_head.shape={model.lm_head.weight.shape}")
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        _print(f"now.embedding.shape={model.model.embed_tokens.weight.shape}")
        _print(f"now.lm_head.shape={model.lm_head.weight.shape}")
    
    if args.freeze_model:
        _print(f"Freezing Model:\nnew_token: {len(special_token_list)}\norigin_length: {len(tokenizer) - len(special_token_list)}")
        model.freeze_embed(
            new_token_cnt=len(special_token_list), 
            origin_length=len(tokenizer) - len(special_token_list)
        )
    else:
        _print("mean ...")
        with torch.no_grad():
            for idx, token in enumerate(reversed(special_token_list), start=1):
                description = special_token_desp_dict[token]
                tokenized = tokenizer.tokenize(description)
                tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)

                # embedding layer
                new_embedding = model.model.embed_tokens.weight[tokenized_ids].mean(axis=0)
                model.model.embed_tokens.weight[-idx, :] = new_embedding.clone().detach().requires_grad_(True)

                # lm_head layer
                last_embedding = model.lm_head.weight[tokenized_ids].mean(axis=0)
                model.lm_head.weight[-idx, :] = last_embedding.clone().detach().requires_grad_(True)
    
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print("Trainable Parameters:")
    for param_name in trainable_params:
        print(param_name)

    return model, tokenizer,hook_handle

def get_dataset_and_data_collator(
    args,
    comp_config:Config,
    tokenizer:Tokenizer,
    padding_config:Dict,
    attention_config:Dict,
    sample_config:Dict,
) -> Tuple[MyDataset, MyDataCollator]:
    
    cache_dir=os.path.join(os.path.dirname(args.train_path),"dataset_cache")
    tokenizer_name = os.path.basename(os.path.normpath(args.tokenizer_path))
    dataset_name = os.path.splitext(os.path.basename(args.train_path))[0]
    cache_filename=f"cache_{tokenizer_name}_{dataset_name}.pt"
    
    dataset = MyDataset(
        file_path=args.train_path,
        config=comp_config,
        tokenizer=tokenizer,
        padding_config=padding_config,
        train_on_input=args.train_on_input,
        change_rope=False,
        output_compress_instruction=args.output_compress_instruction,
        # cache_dir=cache_dir,
        cache_dir=None,
        cache_filename=cache_filename,
        force_preprocess=True,
        local_rank=local_rank,
        use_EPL=args.use_EPL,
    )

    data_collator = MyDataCollator(
        dataset=dataset,
        attention_config=attention_config,
        exclude_continue=args.exclude_continue,
        sample_config=sample_config
    )

    return dataset, data_collator


def main():
    args = get_parser()
    if args.output_compress_instruction == "None":
        args.output_compress_instruction = ""
    print(args)
    
    resume_from_checkpoint = None
    if os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            resume_from_checkpoint = os.path.join(args.output_dir, latest_checkpoint)
            print(f"发现检查点，将从 {resume_from_checkpoint} 恢复训练")

    comp_config = Config.from_file(config_path=args.compress_config)

    model, tokenizer, hook_handle = get_model_and_tokenizer(
        args, comp_config
    )

    sample_config:Dict = dict(
        mode=args.mode,
        hybrid=args.hybrid
    )
    attention_config:Dict = dict(
        diagonal=args.diagonal,
        bi_directional=args.bi_directional,
        see_current=args.see_current,
        prefill_compress=args.prefill_compress,
    )
    padding_config = dict(
        padding_side='right',
        label_padding_id=IGNORE_LABEL_ID,
        input_padding_id=tokenizer.eos_token_id,
        max_length=args.max_length,
        position_ids_padding_id=0,
    )

    dataset, data_collator = get_dataset_and_data_collator(
        args=args, 
        comp_config=comp_config,
        tokenizer=tokenizer,
        padding_config=padding_config,
        attention_config=attention_config,
        sample_config=sample_config,
    )

    training_config = TrainingArguments(
        lr_scheduler_type=args.lr_scheduler_type,
        local_rank=args.local_rank,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=1,
        do_eval=False,
        optim="adamw_torch",
        save_strategy="epoch",      # the default value is step
        save_steps=args.save_steps, # if the strategy is epoch, the save_steps is not used.
        output_dir=args.output_dir,
        save_only_model=False,       # don't save the global_steps
        load_best_model_at_end=False,
        deepspeed=args.deepspeed,
        save_total_limit=1,
        report_to="tensorboard",
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_config,
        data_collator=data_collator,
        callbacks=[SaveTokenizerCallback(tokenizer), MTPLossCallback(loss_buffer)]  # 添加回调
    )

    # 1. 弹出默认的 TensorBoardCallback (它目前排在队列最前面)
    tb_callback = trainer.pop_callback(TensorBoardCallback)

    # 2. 如果成功移除了，再加回来保证顺序正确
    if tb_callback is not None:
        trainer.add_callback(tb_callback)

    # 在加载检查点时使用上下文管理器
    if resume_from_checkpoint and not comp_config.forzen_model_train_mtp:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    # 训练完成，显式移除钩子
    hook_handle.remove()
    print("Hook removed successfully.")
    


if __name__ == '__main__':
    main()