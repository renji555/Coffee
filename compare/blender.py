# =====================================================================
# 原作者："Jiang, Dongfu and Ren, Xiang and Lin, Bill Yuchen"
# 来源：https://github.com/yuchenlin/LLM-Blender
# 原许可证：Apache-2.0（见下方许可证文本）
# 修改说明：提取了比较的部分，将比较多个模型生成结果好坏的部分提取了出来
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

import logging
import torch
import numpy as np
import copy
import json
import os
import importlib
import transformers
from typing import List, Union
from pathlib import Path
from compare_util.config import RankerConfig
from compare_util.utils import (load_ranker,
    RankerDataset,
)
from compare_util.gpt_eval.utils import (
    get_scores_from_cmps,
    get_ranks_from_scores
)
from transformers.utils.hub import TRANSFORMERS_CACHE
from tqdm import tqdm
class Blender:
    def __init__(
        self, 
        ranker_config:RankerConfig=None,
    ):
        self.ranker_config = ranker_config
        
        if self.ranker_config is None:
            logging.warning("No ranker config provided, no ranker loaded, please load ranker first through load_ranker()")
        else:
            ranker_path = self.ranker_config.load_checkpoint
            self.loadranker(ranker_path, **self.ranker_config.to_dict())
        
    def loadranker(self, ranker_path:str, device:str=None, **kwargs):
        """Load ranker from a path
            Supported rankers:
                - llm-blender/pair-ranker
                - llm-blender/pair-reward-model
                - llm-blender/PairRM
                - OpenAssistant/reward-model-deberta-v3-large-v2
                - openbmb/UltraRM-13b
                - berkeley-nest/Starling-RM-7B-alpha
                - Local path, e.g. "/path/to/ranker"
        """
        cache_dir = kwargs.pop("cache_dir", TRANSFORMERS_CACHE)
        cache_dir = Path(cache_dir)
        
        if not os.path.exists(ranker_path):
            if not os.path.exists(cache_dir / ranker_path):
                logging.warning(f"Checkpoint '{ranker_path}' does not exist")
                try:
                    # try hugging face hub
                    logging.warning(f"Try dowloading checkpoint from huggingface hub: {ranker_path}")
                    snapshot_download(ranker_path, local_dir=cache_dir / ranker_path)
                    ranker_path = cache_dir / ranker_path
                    logging.warning(f"Successfully downloaded checkpoint to '{ranker_path}'")
                except Exception as e:
                    # try local path
                    logging.warning(f"Failed to download checkpoint from huggingface hub: {ranker_path}")
                    logging.warning(f"Erorr: {e}")
            else:
                ranker_path = cache_dir / ranker_path
        
        # load ranker config from ranker_path
        ranker_path = Path(ranker_path)
        if os.path.exists(ranker_path / "config.json"):
            with open(ranker_path / "config.json", "r") as f:
                ranker_config_json = json.load(f)
            ranker_config = RankerConfig.from_dict(ranker_config_json)
            ranker_config.load_checkpoint = str(ranker_path)
            ranker_config.cache_dir = cache_dir
            self.ranker_config = ranker_config
        else:
            ranker_config_json = {
                "ranker_type": None,
                "model_type": None,
                "model_name": str(ranker_path),
                "cache_dir": cache_dir,
            }
            ranker_config = RankerConfig.from_dict(ranker_config_json)
            self.ranker_config = ranker_config
        for k, v in kwargs.items():
            setattr(self.ranker_config, k, v)
        if ranker_config.model_name is None:
            ranker_config.model_name = str(ranker_path)
    
        # for other rms    
        if ranker_config.ranker_type not in ["pairranker", "summareranker", "simcls"]:
            # tell from the ranker_path
            if ranker_config.model_name.endswith("OpenAssistant/reward-model-deberta-v3-large-v2"):
                ranker_config.ranker_type = "deberta-rm"
                ranker_config.model_type = "deberta-rm"
            elif ranker_config.model_name.endswith("berkeley-nest/Starling-RM-7B-alpha"):
                ranker_config.ranker_type = "starling-rm"
                ranker_config.model_type = "starling-rm"
            elif ranker_config.model_name.endswith("openbmb/UltraRM-13b"):
                ranker_config.ranker_type = "ultra-rm"
                ranker_config.model_type = "ultra-rm"
            else:
                raise ValueError(f"reward model type {ranker_config.model_name} not supported")
            ranker_config.load_checkpoint = None
            
        self.ranker_config.device = device or self.ranker_config.device 
    
        self.ranker, self.ranker_tokenizer, self.ranker_collator = load_ranker(ranker_config)
        device = self.ranker_config.device
        if device in ["cuda", "mps"] and ranker_config.fp16:
            self.ranker = self.ranker.half()
        else:
            self.ranker = self.ranker.float()
        self.ranker = self.ranker.to(device)
        self.ranker.eval()
        print("Successfully loaded ranker")
    
    def rank(
        self, 
        inputs:List[str], 
        candidates:List[List[str]], 
        instructions:List[str]=None, 
        return_scores:bool=False,
        batch_size:int=8,
        disable_tqdm:bool=False,
        **rank_kwargs
    ):
        if self.ranker is None:
            logging.warning("No ranker loaded, please load ranker first through load_ranker()")
            return None
        assert len(inputs) == len(candidates), "Number of inputs and candidates must be the same"
        assert all([len(c) > 0 for c in candidates]), "Each input must have at least one candidate"
        assert all([len(c) == len(candidates[0]) for c in candidates]), "Number of candidates for each input must be the same"
        collate_fn = copy.copy(self.ranker_collator)
        collate_fn.source_maxlength = rank_kwargs.get("source_max_length", None) or self.ranker_config.source_maxlength
        collate_fn.candidate_maxlength = rank_kwargs.get("candidate_max_length", None) or self.ranker_config.candidate_maxlength
        dataset = RankerDataset(inputs, candidates, instructions=instructions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        scores = []
        with torch.no_grad():
            for batch in tqdm(iter(dataloader), desc="Ranking candidates", disable=(disable_tqdm)):
                batch = {k: v.to(self.ranker_config.device) for k, v in batch.items() if v is not None}
                if self.ranker_config.ranker_type == "pairranker":
                    outputs = self.ranker._full_predict(**batch)
                    preds = outputs['logits'].detach().cpu().numpy()
                    batch_scores = get_scores_from_cmps(preds)
                elif self.ranker_config.ranker_type in ["summareranker", "simcls"]:
                    outputs = self.ranker(**batch)
                    batch_scores = outputs['logits'].detach().cpu().numpy()
                elif self.ranker_config.ranker_type in ["deberta-rm"]:
                    outputs = self.ranker(**batch)
                    batch_scores = outputs.logits.detach().cpu().numpy()
                    batch_scores = batch_scores.squeeze(-1).reshape(-1, len(candidates[0]))
                else:
                    outputs = self.ranker(**batch) # outputs is a list of scores
                    batch_scores = outputs.detach().cpu().numpy()
                    batch_scores = batch_scores.reshape(-1, len(candidates[0]))
                scores.append(batch_scores)
        scores = np.concatenate(scores, axis=0)
        if return_scores:
            return scores
        else:
            return get_ranks_from_scores(scores)


