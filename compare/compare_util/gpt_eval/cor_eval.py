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

import json
import numpy as np
import scipy

def cor_pearson(hypo_ranks, ref_ranks):
    """
    Args:
        hypo_ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
        ref_ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    returns:
        cor: float, the mean correlation coefficient
    """
    if isinstance(hypo_ranks, list):
        hypo_ranks = np.array(hypo_ranks)
    if isinstance(ref_ranks, list):
        ref_ranks = np.array(ref_ranks)
    assert hypo_ranks.shape == ref_ranks.shape
    bz, c = hypo_ranks.shape
    hypo_ranks = hypo_ranks.reshape(bz, c).T
    ref_ranks = ref_ranks.reshape(bz, c).T
    cor = 0
    for i in range(c):
        cor += np.corrcoef(hypo_ranks[i], ref_ranks[i])[0, 1]
    cor /= c
    return cor

def cor_spearman(hypo_ranks, ref_ranks):
    """
    Args:
        hypo_ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
        ref_ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    returns:
        cor: float, the mean of the diagonal elements of the spearman correlation matrix
    """
    if isinstance(hypo_ranks, list):
        hypo_ranks = np.array(hypo_ranks)
    if isinstance(ref_ranks, list):
        ref_ranks = np.array(ref_ranks)
    assert hypo_ranks.shape == ref_ranks.shape
    bz, c = hypo_ranks.shape
    hypo_ranks = hypo_ranks.reshape(bz, c).T
    ref_ranks = ref_ranks.reshape(bz, c).T
    cor = 0
    for i in range(c):
        cor += scipy.stats.spearmanr(hypo_ranks[i], ref_ranks[i]).correlation
    cor /= c
    return cor

            
def cor_spearman_footrule(hypo_ranks, ref_ranks):
    """
    Args:
        hypo_ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
        ref_ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    returns:
        cor: float, the mean of the set of the spearman correlation coefficients
    """
    if isinstance(hypo_ranks, list):
        hypo_ranks = np.array(hypo_ranks)
    if isinstance(ref_ranks, list):
        ref_ranks = np.array(ref_ranks)
    assert hypo_ranks.shape == ref_ranks.shape
    bz, c = hypo_ranks.shape
    hypo_ranks = hypo_ranks.reshape(bz, c)
    ref_ranks = ref_ranks.reshape(bz, c)
    return np.abs(hypo_ranks - ref_ranks).sum(axis=-1).mean()

def cor_set_based(hypo_ranks, ref_ranks):
    """
    Args:
        hypo_ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
        ref_ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
        Each element (i, j) represents the rank of the j-th candidate in the i-th sample
    returns:
        cor: float, correlation between ranks1 and ranks2
    """
    if isinstance(hypo_ranks, list):
        hypo_ranks = np.array(hypo_ranks)
    if isinstance(ref_ranks, list):
        ref_ranks = np.array(ref_ranks)
    assert hypo_ranks.shape == ref_ranks.shape
    bz, c = hypo_ranks.shape
    hypo_ranks = hypo_ranks.reshape(bz, c)
    ref_ranks = ref_ranks.reshape(bz, c)
    sims = np.zeros(bz)
    for i in range(bz):
        hypo_ranked_idx = np.argsort(hypo_ranks[i])
        ref_ranked_idx = np.argsort(ref_ranks[i])
        for set_size in range(1, c+1):
            hypo_set = set(hypo_ranked_idx[:set_size])
            ref_set = set(ref_ranked_idx[:set_size])
            sims[i] += len(hypo_set.intersection(ref_set)) / len(hypo_set.union(ref_set))
        sims[i] /= c
    return sims.mean()

COR_MAPS = {
    "pearson": cor_pearson,
    "spearman": cor_spearman,
    "spearman_footrule": cor_spearman_footrule,
    "set_based": cor_set_based,
}