# =====================================================================
# 以下代码复制自LLM-BLENDER
# 原作者："Jiang, Dongfu and Ren, Xiang and Lin, Bill Yuchen"
# 原项目地址：https://github.com/yuchenlin/LLM-Blender
# 原许可证：Apache-2.0（见下方许可证文本）
# =====================================================================

# 原代码的 Apache-2.0 许可证文本（完整保留）
# Copyright [原版权年份] [原作者姓名/组织]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import logging
import safetensors
from pathlib import Path
from .config import RankerConfig
from typing import List
from .model_util import (
    build_ranker,
    build_tokenizer,
    build_collator,
)


def load_ranker(ranker_config: RankerConfig):
    """Load PairRanker model from config file"""
    tokenizer = build_tokenizer(ranker_config.model_name, cache_dir=ranker_config.cache_dir)
    collator = build_collator(ranker_config.ranker_type, tokenizer,
        ranker_config.source_maxlength, ranker_config.candidate_maxlength,
    )
    ranker = build_ranker(
        ranker_config.ranker_type,
        ranker_config.model_type,
        ranker_config.model_name,
        ranker_config.cache_dir,
        ranker_config,
        tokenizer,
    )
    ranker = ranker.eval()
    if ranker_config.load_checkpoint is not None:
        # load checkpoint from local path
        load_checkpoint = Path(ranker_config.load_checkpoint)
        if load_checkpoint.name == "pytorch_model.bin":
            load_checkpoint = load_checkpoint.parent
        
        if (load_checkpoint/"pytorch_model.bin").exists():
            # pytorch_model.bin
            state_dict = torch.load(load_checkpoint/"pytorch_model.bin", map_location="cpu")
            load_result = ranker.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                logging.warning(f"Missing keys: {load_result.missing_keys}")
            else:
                logging.info(f"Successfully loaded checkpoint from '{load_checkpoint}'")
        elif (load_checkpoint/"model.safetensors").exists():
            # model.safetensors
            load_result = safetensors.torch.load_model(ranker, load_checkpoint/"model.safetensors")
            missing_keys, unexpected_keys = load_result
            if missing_keys:
                logging.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys: {unexpected_keys}")
            if not missing_keys and not unexpected_keys:
                logging.info(f"Successfully loaded checkpoint from '{load_checkpoint}'")
        else:
            raise ValueError(f"Cannot find pytorch_model.bin or model.safetensors in {load_checkpoint}")
        
    return ranker, tokenizer, collator

class RankerDataset(torch.utils.data.Dataset):
    def __init__(self, inputs:List[str], candidates:List[List[str]], instructions:List[str]=None, scores=None):
        self.instructions = instructions
        self.inputs = inputs
        self.candidates = candidates
        self.scores = scores

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        instruction = self.instructions[index] if self.instructions is not None else ""
        input_text = self.inputs[index]
        candidates = self.candidates[index]
        scores = self.scores[index] if self.scores is not None else None
        batch = {
            'index' : index,
            'source' : instruction + input_text,
            'candidates' : candidates,
            'scores' : scores,
        }
        batch = {k: v for k, v in batch.items() if v is not None}
        return batch