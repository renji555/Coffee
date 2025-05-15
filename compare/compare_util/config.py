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

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class RankerConfig:
    ranker_type:str = field(
        default=None,
        metadata={"help": "Ranker type, pairranker or reranker \
                  choices: summareranker, dual, pairranker, other;"},
    )
    model_type:str = field(default=None,
        metadata={"help": "Model type, deberta or roberta or other"}
    )
    model_name:str = field(default=None,
        metadata={"help": "Model name"}
    )
    cache_dir:str = field(default=None,
        metadata={"help": "Cache dir"}
    )
    load_checkpoint:str = field(default=None,
        metadata={"help": "Load checkpoint path"}
    )
    source_maxlength:int = field(default=None,
        metadata={"help": "Max length of the source sequence"}
    )
    candidate_maxlength:int = field(default=None,
        metadata={"help": "Max length of the candidate sequence"}
    )
    n_tasks:int = field(default=1,
        metadata={"help": "Number of tasks"}
    )
    num_pos:int = field(default=1,
        metadata={"help": "Number of positive examples used for training, used for top_bottom and all_pair sampling"}
    )
    num_neg:int = field(default=1,
        metadata={"help": "Number of negative examples used for training, used for top_bottom and all_pair sampling"}
    )
    sub_sampling_mode:str = field(default="all_pair",
        metadata={"help": "Sub sampling mode: top_bottom, all_pair, random, uniform"}
    )
    sub_sampling_ratio:float = field(default=0.5,
        metadata={"help": "Sub sampling ratio, used for random and uniform sampling"}
    )
    loss_type:str = field(default="instructgpt",
        metadata={"help": "Loss type: instructgpt, contrastive"}
    )
    reduce_type:str = field(default="linear",
        metadata={"help": "Reduce type: linear, max, mean"}
    )
    inference_mode:str = field(default="bubble",
        metadata={"help": "Inference mode: bubble, full"}
    )
    drop_out:float = field(default=0.05,
        metadata={"help": "Dropout rate"}
    )
    fp16:bool = field(default=True,
        metadata={"help": "Whether to use fp16"}
    )
    device:str = field(default=None,
        metadata={"help": "Device, cuda or cpu or mps"}
    )




                  
