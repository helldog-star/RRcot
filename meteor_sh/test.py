from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/zhaorunsong/models/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

separators = ['.', ',', '?', '!', ';', ':', ' ', '\t', '\n']

for sep in separators:
    token_ids = tokenizer.encode(sep, add_special_tokens=False)
    print(f"{repr(sep):8} -> {token_ids}")


print("pad_token:", repr(tokenizer.pad_token))
print("pad_token_id:", tokenizer.pad_token_id)