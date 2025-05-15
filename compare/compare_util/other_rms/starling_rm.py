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


import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

## Define the reward model function class

class StarlingRM(nn.Module):
    def __init__(self, pretrained_model, config, tokenizer):
        super().__init__()
        model = pretrained_model
        self.rm_config = config
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = tokenizer
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        
        directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith("model.bin"):
                checkpoint = os.path.join(directory, fpath)
                break
        
        self.load_state_dict(torch.load(checkpoint), strict=False)
        self.eval().requires_grad_(False)

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return torch.tensor(scores)

# ## Load the model and tokenizer

# reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf")
# reward_tokenizer = reward_model.tokenizer
# reward_tokenizer.truncation_side = "left"

# directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
# for fpath in os.listdir(directory):
#     if fpath.endswith(".pt") or fpath.endswith("model.bin"):
#         checkpoint = os.path.join(directory, fpath)
#         break
   
# reward_model.load_state_dict(torch.load(checkpoint), strict=False)
# reward_model.eval().requires_grad_(False)


# ## Define the reward function

# def get_reward(samples):
#     """samples: List[str]"""
#     input_ids = []
#     attention_masks = []
#     encodings_dict = reward_tokenizer(
#         samples,
#         truncation=True,
#         max_length=2048,
#         padding="max_length",
#         return_tensors="pt",
#     ).to(reward_device)
#     input_ids = encodings_dict["input_ids"]
#     attention_masks = encodings_dict["attention_mask"]
#     mbs = reward_batch_size
#     out = []
#     for i in range(math.ceil(len(samples) / mbs)):
#         rewards = reward_model(input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs])
#         out.extend(rewards)
#     return torch.hstack(out)

# ## Inference over test prompts with llama2 chat template

# test_sample = ["<s>[INST] Hello? </s> [/INST] Hi, how can I help you?</s>"] 
# reward_for_test_sample = get_reward(test_sample)
# print(reward_for_test_sample)
