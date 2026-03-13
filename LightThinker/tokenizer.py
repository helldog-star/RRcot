
from transformers import AutoTokenizer
from typing import *
from LightThinker.utils import _print, IGNORE_LABEL_ID
from copy import deepcopy
import numpy as np
import random


class Tokenizer:

    def __init__(
        self,
        tokenizer_path:str,
        bos_token:str,
        eos_token:str,
        special_token_list:List[str]=None,
        add_prefix_space:bool=False,
        change_rope:bool=False,
    ):
        self.change_rope:bool = change_rope
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            add_prefix_space=add_prefix_space,
            mean_resizing=False
        )
        if special_token_list != None:
            self.add_special_token(special_token_list)
        self.bos_token:str = bos_token
        self.eos_token:str = eos_token
        self.tokenizer.add_eos_token = False
        self.tokenizer.add_bos_token = False
    
    def add_special_token(self, special_token_list:List[str]):
        _print("expanding tokenizer ...")
        num_added_tokens = self.tokenizer.add_tokens(
            special_token_list, special_tokens=True
        )
        assert num_added_tokens == len(special_token_list), f"{special_token_list}"
        _print(f"{num_added_tokens} tokens have been added including {special_token_list}")
        self.bos_token_id = None if self.bos_token == None else self.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_token_id = None if self.eos_token == None else self.tokenizer.convert_tokens_to_ids(self.eos_token)
        if self.eos_token_id is None:
            assert self.eos_token is None
        if self.bos_token_id is None:
            assert self.bos_token is None
        return self.tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
    
    def __len__(self):
        return len(self.tokenizer)

    def normal_data_tokenize(
        self,
        structured_input:List[str],
        max_length:int,
        train_on_input:bool=False,
        check_consistency:bool=False
    ) -> Dict:

        final_item:Dict = dict(
            input_ids=list(),
            labels=list(),
            position_ids=list(),
        )

        # 1. Tokenize
        tokenized_input_id_list:List = list()
        for segement in structured_input:
            if segement != "":
                tokenized_input_id_list.append(
                    self.tokenizer(
                        segement, return_tensors=None
                    )['input_ids']
                )
            else:
                # Ensure a form similar to user-assistant-user-assistant
                tokenized_input_id_list.append(list())
        if check_consistency:
            whole_input:str = "".join(structured_input)
            tokenized_whole_input = self.tokenizer(
                whole_input, return_tensors=None,add_special_tokens=False
            )['input_ids']
            tokenized_whole_input_from_segement = list()
            for input_ids in tokenized_input_id_list:
                tokenized_whole_input_from_segement.extend(
                    input_ids
                )
            assert tokenized_whole_input_from_segement == tokenized_whole_input, \
                f"consistency check failed.\n{tokenized_whole_input_from_segement}\n{tokenized_whole_input}\n{whole_input}\n{structured_input}"

        # 2. create labels
        tokenized_label_list = deepcopy(tokenized_input_id_list)
        if not train_on_input:
            for i in range(len(tokenized_label_list)):
                if i % 2 == 0:
                    # mask for user part(role mask)
                    tokenized_label_list[i] = [IGNORE_LABEL_ID] * len(tokenized_label_list[i])

        for input_ids in tokenized_input_id_list:
            final_item['input_ids'].extend(input_ids)
        for labels in tokenized_label_list:
            final_item['labels'].extend(labels)


        # 3. truncate
        final_item['input_ids'] = final_item['input_ids'][0:max_length]
        final_item['labels'] = final_item['labels'][0:max_length]
        final_item['position_ids'] = list(range(len(final_item['input_ids'])))

        return final_item

    def aug_data_tokenize(
        self,
        structured_input:List[List],
        # save, abandoned, compressed 
        structured_input_indicator:List[List[str]],
        n_comp_for_output:int,  
        n_continue_for_output:int,
        n_comp_for_prompt:int,
        n_continue_for_prompt:int,
        mask_label_map:Dict[int, int],
        max_length:int,
        train_on_input:bool=False,
        check_consistency:bool=False,
        recover_mode:bool=False,
        use_EPL:bool=False,
        output_comp_adaptive_num_token:List[int]=[],
    ) -> Tuple[List[Dict], Dict]:
        # 1. tokenize
        whole_input = ""
        tokenized_whole_input_from_segement = list()
        
        tokenized_input_id_list:List[List[List[int]]] = list()
        for segement_list in structured_input:
            tokenized_input_id_list.append(list())
            for segement in segement_list:
                if not isinstance(segement, str):
                    assert isinstance(segement, list)
                    tokenized_input_id_list[-1].append(segement)
                    whole_input += self.tokenizer.decode(segement)
                    tokenized_whole_input_from_segement.extend(segement)
                else:
                    tokenized_input_id_list[-1].append(
                        self.tokenizer(segement, return_tensors=None, add_special_tokens=False)['input_ids']
                    )
                    tokenized_whole_input_from_segement.extend(
                        tokenized_input_id_list[-1][-1]
                    )
                    whole_input += segement
        if check_consistency:
            tokenized_whole_input = self.tokenizer(
                whole_input, return_tensors=None, add_special_tokens=False
            )['input_ids']
            assert tokenized_whole_input_from_segement == tokenized_whole_input, \
                f"consistency check failed.\n{tokenized_whole_input_from_segement}\n{tokenized_whole_input}\n{whole_input}\n{structured_input}\n\n\n\n`{self.tokenizer.decode(tokenized_whole_input_from_segement)}`\n`{self.tokenizer.decode(tokenized_whole_input)}`"

        # 2. create label
        tokenized_label_list = deepcopy(tokenized_input_id_list)
        if not train_on_input:
            for i in range(len(tokenized_label_list)):
                for j in range(len(tokenized_label_list[i])):
                    if i % 2 == 0:
                        # mask for user's part
                        tokenized_label_list[i][j] = [IGNORE_LABEL_ID] * len(tokenized_label_list[i][j])
                    else:
                        for k in range(len(tokenized_label_list[i][j])):
                            if tokenized_label_list[i][j][k] in mask_label_map:
                                tokenized_label_list[i][j][k] = mask_label_map[
                                    tokenized_label_list[i][j][k]
                                ]
        else:
            for i in range(len(tokenized_label_list)):
                for j in range(len(tokenized_label_list[i])):
                    for k in range(len(tokenized_label_list[i][j])):
                            if tokenized_label_list[i][j][k] in mask_label_map:
                                tokenized_label_list[i][j][k] = mask_label_map[
                                    tokenized_label_list[i][j][k]
                                ]

        # 3. final results
        final_item:Dict = dict(
            input_ids=list(),
            labels=list(),
            locate_index=list(),
            position_ids=list(),
            locate_indicator=list(),
            system_prompt_length=list()
        )
        compression_count = 0
        adaptive_index = 0
        for i in range(len(tokenized_label_list)):
            if len(final_item['input_ids']) >= max_length:
                break
            if i == 0:
                final_item["system_prompt_length"].append(len(tokenized_input_id_list[0][0]))
            for j in range(len(tokenized_label_list[i])):
                if len(final_item['input_ids']) >= max_length:
                    break
                assert structured_input_indicator[i][j] in [
                    'save', 'abandoned', 'compressed-prompt', 'compressed-output'
                ]
                if structured_input_indicator[i][j] == 'abandoned':
                    # print(structured_input_indicator[i][j+1])
                    if j+1 < len(structured_input_indicator[i]):

                        assert structured_input_indicator[i][j+1] in ['compressed-prompt', 'compressed-output']
                        n_comp = n_comp_for_prompt if structured_input_indicator[i][j+1] == 'compressed-prompt' else n_comp_for_output
                        n_continue = n_continue_for_prompt if structured_input_indicator[i][j+1] == 'compressed-prompt' else n_continue_for_output
                        if len(output_comp_adaptive_num_token) > 0:
                            n_comp = output_comp_adaptive_num_token[adaptive_index]
                            adaptive_index += 1
                        assert len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue >= 0
                        # mask for instruction
                        if len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue > 0:
                            tokenized_label_list[i][j+1][0:len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue] = [IGNORE_LABEL_ID] * (len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue)
                        final_item['locate_indicator'].append(structured_input_indicator[i][j+1])
                        final_item['locate_index'].append(
                            [
                                len(final_item['input_ids']), 
                                len(final_item['input_ids']) + len(tokenized_input_id_list[i][j]),
                                len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue,
                                n_comp,
                                n_continue
                            ]
                        )

                        # 如果是 compressed-output, 生成均匀分布 position_ids，compressed-output部分只包含compressed-token和continue-token
                        if use_EPL and structured_input_indicator[i][j+1] == 'compressed-output':
                            n_abandoned = len(tokenized_input_id_list[i][j])
                            n_compressed = n_comp

                            base_pos = len(final_item['input_ids']) - compression_count
                            end_pos = len(final_item['input_ids']) + n_abandoned - compression_count
                            compressed_positions = []
                            step = n_abandoned / n_compressed
                            for k in range(n_compressed):
                                # k * step: 当前分段的起始
                                # step / 2: 当前分段的中心偏移量
                                # base_pos: 全局起始偏移
                                # int(...): 向下取整得到整数索引
                                center_offset = int(k * step + step / 2)
                                
                                # 计算最终位置
                                pos = base_pos + center_offset
                                compressed_positions.append(pos)
                            
                        else:
                            compressed_positions = None
                
                if use_EPL and structured_input_indicator[i][j] == 'compressed-output':
                    compression_count += n_compressed
                    for k in range(len(tokenized_label_list[i][j])):
                        if len(final_item['input_ids']) >= max_length:
                            break
                        # position_id 逻辑
                        if structured_input_indicator[i][j] == 'compressed-output' and compressed_positions is not None and n_compressed != 0:
                            # 为了处理continue token
                            n_compressed = n_compressed - 1
                            final_item['position_ids'].append(compressed_positions[k])
                        else:
                            final_item['position_ids'].append(end_pos)
                        
                        final_item['input_ids'].append(tokenized_input_id_list[i][j][k])
                        final_item['labels'].append(tokenized_label_list[i][j][k])
                else:
                    for k in range(len(tokenized_label_list[i][j])):
                        if len(final_item['input_ids']) >= max_length:
                            break
                        final_item['position_ids'].append(len(final_item['input_ids']) - compression_count)
                        final_item['input_ids'].append(
                            tokenized_input_id_list[i][j][k]
                        )
                        final_item['labels'].append(
                            tokenized_label_list[i][j][k]
                        )
                # print(self.tokenizer.decode(final_item['input_ids']))

        # 4. recover
        # we do not use revover mode
        recover_item_list:List[Dict] = list()
        if recover_mode:
            # we do not use recover mode 
            delta_length = max_length - len(final_item['input_ids']) - 2
            if delta_length <= 0:
                return recover_item_list, final_item
            for idx, index_tuple in enumerate(final_item['locate_index']):
                start, end, inst_len, n_comp, n_continue = index_tuple
                if final_item['locate_indicator'][idx] == 'compressed-prompt':
                    continue
                else:
                    new_item = dict(
                        input_ids=final_item['input_ids'][start:end],
                        labels=final_item['labels'][start:end],
                        corresp_attention=[end, inst_len, n_comp, n_continue],
                        indicator=final_item['locate_indicator'][idx]
                    )
                    delta_length -= (end-start)
                    if delta_length < 0:
                        break
                    recover_item_list.append(new_item)
                
        return recover_item_list, final_item


    def _insert_register_token(self, input_ids: list[int], register_token_id: int) -> list[int]:
        # 在每个token前面插入一个register token
        new_input_ids = []
        for i in range(len(input_ids)):
            new_input_ids.append(register_token_id)
            new_input_ids.append(input_ids[i])
        return new_input_ids
    
    def _calculate_position_ids_with_register(self, input_ids: list[int], register_token_id: int, offset, base_pos: int) -> list[int]:
        # input_ids为[R,C,R,C,...,R,C]的形式
        # 找到所有register_token_id（R）对应的位置
        # 对C从base_pos开始，每隔一个位置插入一个位置编码，直到最后一个C
        # 对R分配的位置编码为R后面第offset-1个C的位置编码（offset为1时取紧邻的后面的C，offset为0时取紧邻的前面的C）
        position_ids = []
        n_content = len(input_ids) // 2  # C的总数量
        content_pos = base_pos  # 当前C要分配的位置编码
        for i in range(len(input_ids)):
            if input_ids[i] == register_token_id:
                # R: 分配R后面第(offset-1)个C的位置编码
                # R在索引2k处，对应第k个R-C对，第(offset)个C的content index为k+offset-1
                k = i // 2
                target_c_idx = k + offset - 1
                if target_c_idx >= n_content:
                    pos_id = base_pos + n_content - 1  # 越界则用最后一个C的位置
                else:
                    pos_id = base_pos + target_c_idx
                position_ids.append(pos_id)
            else:
                # C: 从base_pos开始递增分配
                position_ids.append(content_pos)
                content_pos += 1
        return position_ids
    
    def aug_data_tokenize_apa_mtp(
        self,
        structured_input:List[List],
        # save, abandoned, compressed 
        structured_input_indicator:List[List[str]],
        n_comp_for_output:int,  
        n_continue_for_output:int,
        n_comp_for_prompt:int,
        n_continue_for_prompt:int,
        mask_label_map:Dict[int, int],
        max_length:int,
        train_on_input:bool=False,
        check_consistency:bool=False,
        recover_mode:bool=False,
        use_EPL:bool=False,
        regitser_token:str = None,
        output_comp_adaptive_num_token:List[int]=[],
    ) -> Tuple[List[Dict], Dict]:

        # 1. tokenize
        whole_input = ""
        tokenized_whole_input_from_segement = list()
        tokenized_input_id_list:List[List[List[int]]] = list()

        # register token id
        regitser_token_id = self.tokenizer.convert_tokens_to_ids(regitser_token)
        # mtp register offset
        register_mtp_offset = random.randint(1, 4) 
        
        for i, segement_list in enumerate(structured_input):
            tokenized_input_id_list.append(list())
            for j, segement in enumerate(segement_list):

                # if not isinstance(segement, str):
                #     assert isinstance(segement, list)
                #     tokenized_input_id_list[-1].append(segement)
                #     whole_input += self.tokenizer.decode(segement)
                #     tokenized_whole_input_from_segement.extend(segement)
                # else:
                #     tokenized_input_id_list[-1].append(
                #         self.tokenizer(segement, return_tensors=None)['input_ids']
                #     )
                #     tokenized_whole_input_from_segement.extend(
                #         tokenized_input_id_list[-1][-1]
                #     )
                #     whole_input += segement

                if structured_input_indicator[i][j] == 'abandoned':
                    tokenized_input_id = self.tokenizer(segement, return_tensors=None, add_special_tokens=False)['input_ids']
                    # 插入register tokens
                    tokenized_input_id_add_register = self._insert_register_token(tokenized_input_id, regitser_token_id)
                    tokenized_input_id_list[-1].append(tokenized_input_id_add_register)
                    tokenized_whole_input_from_segement.extend(
                        tokenized_input_id_list[-1][-1]
                    )
                    whole_input += segement
                else:
                    tokenized_input_id_list[-1].append(
                        self.tokenizer(segement, return_tensors=None, add_special_tokens=False)['input_ids']
                    )
                    tokenized_whole_input_from_segement.extend(
                        tokenized_input_id_list[-1][-1]
                    )
                    whole_input += segement
        
        # we do not check consistency for register token    
        # if check_consistency:
        #     tokenized_whole_input = self.tokenizer(
        #         whole_input, return_tensors=None
        #     )['input_ids']
        #     assert tokenized_whole_input_from_segement == tokenized_whole_input, \
        #         f"consistency check failed.\n{tokenized_whole_input_from_segement}\n{tokenized_whole_input}\n{whole_input}\n{structured_input}\n\n\n\n`{self.tokenizer.decode(tokenized_whole_input_from_segement)}`\n`{self.tokenizer.decode(tokenized_whole_input)}`"

        # 2. create label
        tokenized_label_list = deepcopy(tokenized_input_id_list)
        if not train_on_input:
            for i in range(len(tokenized_label_list)):
                for j in range(len(tokenized_label_list[i])):
                    if i % 2 == 0:
                        # mask for user's part
                        tokenized_label_list[i][j] = [IGNORE_LABEL_ID] * len(tokenized_label_list[i][j])
                    else:
                        for k in range(len(tokenized_label_list[i][j])):
                            if tokenized_label_list[i][j][k] in mask_label_map:
                                tokenized_label_list[i][j][k] = mask_label_map[
                                    tokenized_label_list[i][j][k]
                                ]
                            if tokenized_label_list[i][j][k] == regitser_token_id:
                                if k+1+2*register_mtp_offset < len(tokenized_label_list[i][j]):
                                    tokenized_label_list[i][j][k] = tokenized_label_list[i][j][k+1+2*register_mtp_offset]
                                else:
                                    # 越界的部分 label 设置为 IGNORE_LABEL_ID
                                    tokenized_label_list[i][j][k] = IGNORE_LABEL_ID
        
        else:
            for i in range(len(tokenized_label_list)):
                for j in range(len(tokenized_label_list[i])):
                    for k in range(len(tokenized_label_list[i][j])):
                            if tokenized_label_list[i][j][k] in mask_label_map:
                                tokenized_label_list[i][j][k] = mask_label_map[
                                    tokenized_label_list[i][j][k]
                                ]

        # 3. final results
        final_item:Dict = dict(
            input_ids=list(),
            labels=list(),
            locate_index=list(),
            position_ids=list(),
            locate_indicator=list(),
            system_prompt_length=list()
        )
        compression_count = 0
        register_count = 0
        adaptive_index = 0

        for i in range(len(tokenized_label_list)):
            # 加入register token后，input_ids长度约为原来的2倍，将max_length设置为baseline的2倍
            if len(final_item['input_ids']) >= max_length:
                break
            if i == 0:
                final_item["system_prompt_length"].append(len(tokenized_input_id_list[0][0]))
            for j in range(len(tokenized_label_list[i])):
                if len(final_item['input_ids']) >= max_length:
                    break
                assert structured_input_indicator[i][j] in [
                    'save', 'abandoned', 'compressed-prompt', 'compressed-output'
                ]
                if structured_input_indicator[i][j] == 'abandoned':
                    # print(structured_input_indicator[i][j+1])
                    if j+1 < len(structured_input_indicator[i]):

                        assert structured_input_indicator[i][j+1] in ['compressed-prompt', 'compressed-output']
                        n_comp = n_comp_for_prompt if structured_input_indicator[i][j+1] == 'compressed-prompt' else n_comp_for_output
                        n_continue = n_continue_for_prompt if structured_input_indicator[i][j+1] == 'compressed-prompt' else n_continue_for_output
                        if len(output_comp_adaptive_num_token) > 0:
                            n_comp = output_comp_adaptive_num_token[adaptive_index]
                            adaptive_index += 1
                        assert len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue == 0
                        # apa_mtp just support compressed-output, so we do not mask for instruction
                        # # mask for instruction
                        # if len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue > 0:
                        #     tokenized_label_list[i][j+1][0:len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue] = [IGNORE_LABEL_ID] * (len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue)
                        final_item['locate_indicator'].append(structured_input_indicator[i][j+1])
                        # final_item['locate_index']的obj分别为
                        # 被压缩cot step的起始位置和结束位置（是左闭右开区间）# 0
                        # 被压缩cot step的compressed-token和continue-token数量
                        final_item['locate_index'].append(
                            [
                                len(final_item['input_ids']), 
                                len(final_item['input_ids']) + len(tokenized_input_id_list[i][j]),
                                len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue,
                                n_comp,
                                n_continue
                            ]
                        )

                        # 如果是 compressed-output, 生成均匀分布 position_ids，compressed-output部分只包含compressed-token和continue-token
                        if use_EPL and structured_input_indicator[i][j+1] == 'compressed-output':
                            # 因为加入了register token，所以需要//2
                            n_abandoned = len(tokenized_input_id_list[i][j]) // 2
                            n_compressed = n_comp
                            
                            # 真实的位置编码是相对于register token的，所以需要-register_count
                            base_pos = len(final_item['input_ids']) - compression_count - register_count
                            end_pos = len(final_item['input_ids']) + n_abandoned - compression_count - register_count
                            compressed_positions = []
                            step = n_abandoned / n_compressed
                            for k in range(n_compressed):
                                # k * step: 当前分段的起始
                                # step / 2: 当前分段的中心偏移量
                                # base_pos: 全局起始偏移
                                # int(...): 向下取整得到整数索引
                                center_offset = int(k * step + step / 2)
                                
                                # 计算最终位置
                                pos = base_pos + center_offset
                                compressed_positions.append(pos)
                        else:
                            n_abandoned = len(tokenized_input_id_list[i][j]) // 2
                            compressed_positions = None
                
                if use_EPL and structured_input_indicator[i][j] == 'compressed-output':
                    compression_count += n_compressed
                    for k in range(len(tokenized_label_list[i][j])):
                        if len(final_item['input_ids']) >= max_length:
                            break
                        # position_id 逻辑
                        if structured_input_indicator[i][j] == 'compressed-output' and compressed_positions is not None and n_compressed != 0:
                            # 为了处理continue token
                            n_compressed = n_compressed - 1
                            final_item['position_ids'].append(compressed_positions[k])
                        else:
                            final_item['position_ids'].append(end_pos)
                        
                        final_item['input_ids'].append(tokenized_input_id_list[i][j][k])
                        final_item['labels'].append(tokenized_label_list[i][j][k])
                else:
                    # 在这里处理加入register token后abandoned的情况
                    if structured_input_indicator[i][j] == 'abandoned':
                        segment_input_ids = tokenized_input_id_list[i][j]
                        base_pos = len(final_item['input_ids']) - compression_count - register_count
                        position_ids_with_register = self._calculate_position_ids_with_register(
                            segment_input_ids, regitser_token_id, register_mtp_offset, base_pos
                        )
                        for k in range(len(tokenized_label_list[i][j])):
                            if len(final_item['input_ids']) >= max_length:
                                break

                            final_item['input_ids'].append(segment_input_ids[k])
                            final_item['labels'].append(tokenized_label_list[i][j][k])
                            final_item['position_ids'].append(position_ids_with_register[k])
                        register_count += n_abandoned
                    # 处理save和非EPL情况下的compressed-output
                    else:
                        for k in range(len(tokenized_label_list[i][j])):
                            if len(final_item['input_ids']) >= max_length:
                                break
                            final_item['position_ids'].append(len(final_item['input_ids']) - compression_count - register_count)
                            
                            final_item['input_ids'].append(tokenized_input_id_list[i][j][k])
                            final_item['labels'].append(tokenized_label_list[i][j][k])

        # 4. recover
        # we do not use revover mode
        recover_item_list:List[Dict] = list()
       
        return recover_item_list, final_item
  

if __name__ == '__main__':
    pass
    


