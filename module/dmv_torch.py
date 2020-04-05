import os
from typing import Union, Tuple
from dataclasses import dataclass

import cupyx as cpx
import torch
from utils.common import *
from utils.functions import torch2cp, cp2torch
from torch import nn
from utils.options import Options
from module.eisner_v2 import batch_inside


@dataclass
class DMVOptions(Options):
    num_tag: int = 0
    max_len: int = 10  # max len for all dataset
    cv: int = 1  # 1 or 2
    e_step_mode: str = 'em'  # em or viterbi
    count_smoothing: float = 1e-1  # smooth for init_param
    param_smoothing: float = 1e-1  # smooth for m_step


class DMV(nn.Module):
    """Dependency Model with Valence

    Note: there is randomness when using scatter_add because it uses atomicAdd
    """

    def __init__(self, o: DMVOptions):
        super().__init__()
        self.o = o
        assert o.e_step_mode in ('em', 'viterbi')
        self.initializing = True  # enable ndmv support if False

        # if self.initializing is False, this param will be used
        # instead of self.trans_param and self.dec_param.
        self.all_trans_param = None
        self.all_dec_param = None

        # root_param: child_tag
        # trans_param: head_tag, child_tag, direction, cv
        # dec_param: head_tag, direction, dv, decision
        self.root_param = nn.Parameter(torch.FloatTensor(o.num_tag))
        self.trans_param = nn.Parameter(torch.FloatTensor(o.num_tag, o.num_tag, 2, o.cv))
        self.dec_param = nn.Parameter(torch.FloatTensor(o.num_tag, 2, 2, 2))

        self.function_mask_set = None  # for UD

        self.trans_buffer = None  # pinned memory for faster CPU-GPU transfer
        self.dec_buffer = None

    def parse(self, id_array, tag_array, len_array):
        tag_array = self.input_gaurd(tag_array)
        trans_scores, dec_scores = self.get_scores(id_array, tag_array, len_array)
        heads, *_ = batch_parse(trans_scores, dec_scores, len_array)

        heads = heads.get()
        parse_results = {id_array[i]: heads[i] for i in range(id_array.shape[0])}
        return parse_results

    def forward(self, id_array, tag_array, len_array):
        batch_size, real_len = tag_array.shape
        mode = 'sum' if self.o.e_step_mode == 'em' else 'max'

        tag_array = self.input_gaurd(tag_array)
        trans_scores, dec_scores = self.get_scores(id_array, tag_array, len_array)
        prob = torch.nn.functional.one_hot(tag_array, self.o.num_tag).to(torch.float)
        ictable, iitable, prob = batch_inside(trans_scores, dec_scores, prob, len_array, mode)
        return prob

    def init_param(self, dataset, getter=None):
        # require same_len
        harmonic_sum = [0., 1.]

        dec_param = torch2cp(self.dec_param.data)
        root_param = torch2cp(self.root_param.data)
        trans_param = torch2cp(self.trans_param.data)
        dec_param.fill(0.)
        root_param.fill(0.)
        trans_param.fill(0.)

        def get_harmonic_sum(n):
            nonlocal harmonic_sum
            while n >= len(harmonic_sum):
                harmonic_sum.append(harmonic_sum[-1] + 1 / len(harmonic_sum))
            return harmonic_sum[n]

        def update_decision(change, norm_counter, pos_array):
            _, word_num = pos_array.shape
            for i in range(word_num):
                pos = pos_array[:, i]
                for direction in range(2):
                    if change[i, direction] > 0:
                        # + and - are just for distinguish, see self.first_child_update
                        cpx.scatter_add(norm_counter, (pos, direction, NOCHILD, GO), 1)
                        cpx.scatter_add(norm_counter, (pos, direction, HASCHILD, GO), -1)
                        cpx.scatter_add(dec_param, (pos, direction, HASCHILD, GO), change[i, direction])
                        cpx.scatter_add(norm_counter, (pos, direction, NOCHILD, STOP), -1)
                        cpx.scatter_add(norm_counter, (pos, direction, HASCHILD, STOP), 1)
                        cpx.scatter_add(dec_param, (pos, direction, NOCHILD, STOP), 1)
                    else:
                        cpx.scatter_add(dec_param, (pos, direction, NOCHILD, STOP), 1)

        def first_child_update(norm_counter):
            all_param = dec_param.flatten()
            all_norm = norm_counter.flatten()
            mask = (all_param <= 0) | (0 <= all_norm)
            ratio = - all_param / all_norm
            ratio[mask] = 1.
            return cp.min(ratio)

        # shape same as self.dec_param
        norm_counter = cpfzeros((self.o.num_tag, 2, 2, 2))
        change = cpfzeros((self.o.max_len, 2))

        for arrays in dataset.batch_data:
            if getter:
                pos_array = getter(arrays)
            else:
                pos_array = cpasarray(arrays[1])

            batch_size, word_num = pos_array.shape
            change.fill(0.)
            cpx.scatter_add(root_param, pos_array.flatten(), 1. / word_num)
            if word_num > 1:
                for child_i in range(word_num):
                    child_sum = get_harmonic_sum(child_i - 0) + get_harmonic_sum(word_num - child_i - 1)
                    scale = (word_num - 1) / word_num / child_sum
                    for head_i in range(word_num):
                        if child_i == head_i:
                            continue
                        direction = 0 if head_i > child_i else 1
                        head_pos = pos_array[:, head_i]
                        child_pos = pos_array[:, child_i]
                        diff = scale / abs(head_i - child_i)
                        cpx.scatter_add(trans_param, (head_pos, child_pos, direction), diff)
                        change[head_i, direction] += diff
            update_decision(change, norm_counter, pos_array)

        trans_param += self.o.count_smoothing
        dec_param += self.o.count_smoothing
        root_param += self.o.count_smoothing

        es = first_child_update(norm_counter)
        norm_counter *= 0.9 * es
        dec_param += norm_counter

        root_param_sum = cp.sum(root_param)
        trans_param_sum = cp.sum(trans_param, axis=1, keepdims=True)
        decision_param_sum = cp.sum(dec_param, axis=3, keepdims=True)

        root_param /= root_param_sum
        trans_param /= trans_param_sum
        dec_param /= decision_param_sum

        cp.log(trans_param, out=trans_param)
        cp.log(root_param, out=root_param)
        cp.log(dec_param, out=dec_param)

    def init_pretrained(self, dataset, getter=None):
        if getter is None:
            def getter(x):
                return cpasarray(x[1])

        def recovery_one(heads):
            left_most = np.arange(len(heads))
            right_most = np.arange(len(heads))
            for idx, each_head in enumerate(heads):
                if each_head in (0, len(heads) + 1):  # skip head is ROOT
                    continue
                else:
                    each_head -= 1
                assert each_head >= 0
                if idx < left_most[each_head]:
                    left_most[each_head] = idx
                if idx > right_most[each_head]:
                    right_most[each_head] = idx

            valences = npiempty((len(heads), 2))
            head_valences = npiempty(len(heads))

            for idx, each_head in enumerate(heads):
                if each_head in (0, len(heads) + 1):
                    valences[idx] = HASCHILD if len(heads) > 1 else NOCHILD
                    continue
                else:
                    each_head -= 1
                valences[idx, 0] = NOCHILD if left_most[idx] == idx else HASCHILD
                valences[idx, 1] = NOCHILD if right_most[idx] == idx else HASCHILD
                if each_head > idx:  # d = 0
                    head_valences[idx] = NOCHILD if left_most[each_head] == idx else HASCHILD
                else:
                    head_valences[idx] = NOCHILD if right_most[each_head] == idx else HASCHILD
            return valences, head_valences

        heads = npiempty((len(dataset), self.o.max_len + 1))
        valences = npiempty((len(dataset), self.o.max_len + 1, 2))
        head_valences = npiempty((len(dataset), self.o.max_len + 1))

        for idx, instance in enumerate(dataset.instances):
            one_heads = npasarray(list(map(int, instance.misc)))
            one_valences, one_head_valences = recovery_one(one_heads)
            heads[idx, 1:instance.len + 1] = one_heads
            valences[idx, 1:instance.len + 1] = one_valences
            head_valences[idx, 1:instance.len + 1] = one_head_valences

        heads = cpasarray(heads)
        valences = cpasarray(valences)
        head_valences = cpasarray(head_valences)

        batch_size, sentence_len = heads.shape
        len_array = cpasarray(dataset.get_len())
        batch_arange = cp.arange(batch_size)

        save_batch_data = dataset.batch_data
        dataset.build_batchs(len(dataset), same_len=False, shuffle=False)
        tag_array = getter(dataset.batch_data[0])
        dataset.batch_data = save_batch_data

        self.reset_root_counter()
        self.batch_trans_trace = cpfzeros((batch_size, self.o.max_len, self.o.max_len, 2, self.o.cv))
        self.batch_dec_trace = cpfzeros((batch_size, self.o.max_len, self.o.max_len, 2, 2, 2))

        for m in range(1, sentence_len):
            h = heads[:, m]
            direction = (h <= m).astype(cpi)
            h_valence = head_valences[:, m]
            m_valence = valences[:, m]
            m_child_valence = h_valence if self.o.cv > 1 else cp.zeros_like(h_valence)

            len_mask = ((h <= len_array) & (m <= len_array))

            self.batch_dec_trace[batch_arange, m - 1, m - 1, 0, m_valence[:, 0], STOP] = len_mask
            self.batch_dec_trace[batch_arange, m - 1, m - 1, 1, m_valence[:, 1], STOP] = len_mask

            head_mask = h == 0
            mask = head_mask * len_mask
            if mask.any():
                cpx.scatter_add(self.root_counter, tag_array[:, m - 1], mask)

            head_mask = ~head_mask
            mask = head_mask * len_mask
            if mask.any():
                self.batch_trans_trace[batch_arange, h - 1, m - 1, direction, m_child_valence] = mask
                self.batch_dec_trace[batch_arange, h - 1, m - 1, direction, h_valence, GO] = mask

        d, t = self.get_batch_counter_by_tag(tag_array, mode=1)
        self.m_step(t[0], d[0])

    def set_function_mask(self, functions_to_mask):
        self.function_mask_set = cpasarray(functions_to_mask)

    def function_mask(self, batch_scores, batch_pos):
        batch_size, sentence_length, *_ = batch_scores.shape
        in_mask = cp.in1d(batch_pos, self.function_mask_set).reshape(
            batch_pos.shape)
        batch_scores[in_mask] = -1e5

    def get_batch_counter_by_tag(self, tag_array, num_tag=None, mode=0):
        """counter[batch, sentence_len, ...] to counter[batch, num_tag, ...]

        mode=0, sum in sentence
        mode=1, sum over batch
        """
        if self.batch_trans_trace is None or self.batch_dec_trace is None:
            raise ValueError("No trace can be used")

        batch_size, max_len = tag_array.shape
        if num_tag is None:
            num_tag = self.o.num_tag

        dec_post_dim = (2, 2, 2)
        if mode == 0:
            dec_out = cpfzeros((batch_size, num_tag, *dec_post_dim))
            sentence_id = cp.tile(cp.arange(batch_size).reshape(batch_size, 1), (1, max_len))
            index = (sentence_id.flatten(), tag_array.flatten())
        else:
            dec_out = cpfzeros((1, num_tag, *dec_post_dim))
            index = (0, tag_array.flatten(),)
        cpx.scatter_add(dec_out, index, cp.sum(self.batch_dec_trace, 2).reshape(-1, *dec_post_dim))

        trans_post_dim = (2, self.o.cv)
        head_ids = cp.tile(cp.expand_dims(tag_array, 2), (1, 1, max_len))
        child_ids = cp.tile(cp.expand_dims(tag_array, 1), (1, max_len, 1))
        if mode == 0:
            trans_out = cpfzeros((batch_size, num_tag, num_tag, *trans_post_dim))
            sentence_id = cp.tile(sentence_id, (1, max_len))
            index = (sentence_id.flatten(), head_ids.flatten(), child_ids.flatten())
        else:
            trans_out = cpfzeros((1, num_tag, num_tag, *trans_post_dim))
            index = (0, head_ids.flatten(), child_ids.flatten())
        cpx.scatter_add(trans_out, index, self.batch_trans_trace.reshape(-1, *trans_post_dim))
        return dec_out, trans_out

    def get_scores(self, id_array, tag_array, len_array):
        """ build scores matrices for dmv.

        Masks:
          1. NO mask for length
          2. mask NON-ROOT to ROOT transition scores.
          3. we set ROOT`s decision scores to log(1).
        """
        if self.initializing:
            return self.evaluate_batch_score(tag_array)
        else:
            return self.collect_scores(id_array, tag_array, len_array)

    def evaluate_batch_score(self, tag_array):
        """param[tag_num, ...] to param[batch, sentence_len, ...]"""
        # trans_scores:     batch, head, ntag, child, ntag, cv
        # dec_scores:       batch, head, ntag, direction, dv, decision
        batch_size, fake_len = tag_array.shape
        ntag, cv = self.o.num_tag, self.o.cv

        trans_scores = self.trans_param.view(1, ntag, 1, ntag, 2, cv).repeat(fake_len, 1, fake_len, 1, 1, 1)
        d_indexer = (1 - np.tri(fake_len, k=-1, dtype=int)).reshape(fake_len, 1, fake_len, 1, 1, 1)
        d_indexer = torch.tensor(d_indexer, device='cuda').expand(-1, ntag, -1, ntag, -1, cv)
        trans_scores = trans_scores.gather(4, d_indexer).squeeze(4)

        trans_scores[0] = self.root_param.view(1, -1, 1)
        trans_scores[:, :, 0] = -1e30
        trans_scores = trans_scores.unsqueeze(0).expand(batch_size, *trans_scores.shape)

        dec_scores = self.dec_param.view(1, 1, *self.dec_param.shape)\
            .expand(batch_size, fake_len, *self.dec_param.shape)

        return trans_scores, dec_scores

    def collect_scores(self, idx_array, tag_array, len_array):
        batch_size = idx_array.shape[0]
        max_len = np.max(len_array)

        if self.o.use_gpu_param:
            trans_scores = self.all_trans_param[idx_array]
            dec_scores = self.all_dec_param[idx_array]
        else:
            if self.trans_buffer is None:
                self.dec_buffer = cp.cuda.alloc_pinned_memory(
                    batch_size * (self.o.max_len + 1) * 8 * (np.dtype(npf).itemsize))
                self.trans_buffer = cp.cuda.alloc_pinned_memory(
                    batch_size * (self.o.max_len + 1) * (self.o.max_len + 1) * self.o.cv * (np.dtype(npf).itemsize))
            trans_scores = np.frombuffer(self.trans_buffer, npf, batch_size * (max_len + 1) * (max_len + 1) * self.o.cv)\
                .reshape(batch_size, (max_len + 1), (max_len + 1), self.o.cv)
            dec_scores = np.frombuffer(self.dec_buffer, npf, batch_size * (max_len + 1) * 8) \
                .reshape(batch_size, (max_len + 1), 2, 2, 2)

            if DEBUG:
                trans_scores.fill(10000.)
                dec_scores.fill(10000.)  # check mask leak

            for i, idx in enumerate(idx_array):
                trans_scores[i, 1:len_array[i] + 1, 1:len_array[i] + 1] = self.all_trans_param[idx]
                dec_scores[i, 1:len_array[i] + 1] = self.all_dec_param[idx]

        trans_scores = cpasarray(trans_scores)
        root_param = cp.expand_dims(self.root_param, 0)
        root_scores = cp.expand_dims(cp.take_along_axis(root_param, tag_array, 1), -1)
        trans_scores[:, 0, :, :] = root_scores
        trans_scores[:, :, 0, :] = -cp.inf

        dec_scores = cpasarray(dec_scores)
        dec_scores[:, 0] = 0

        return trans_scores, dec_scores

    def put_decision_param(self, id_array, param, len_array):
        if self.o.use_gpu_param:
            self.all_dec_param[id_array, 1:] = param
        else:
            for i, idx in enumerate(id_array):
                self.all_dec_param[idx][:] = param[i, :len_array[i]]
                assert not DEBUG or not np.isnan(self.all_dec_param[idx]).any()

    def put_transition_param(self, id_array, param, len_array):
        if self.o.use_gpu_param:
            self.all_trans_param[id_array, 1:, 1:] = param
        else:
            for i, idx in enumerate(id_array):
                self.all_trans_param[idx][:] = param[i, :len_array[i], :len_array[i]]
                assert not DEBUG or not np.isnan(self.all_trans_param[idx]).any()

    def reset_root_counter(self):
        self.root_counter.fill(0.)

    def init_specific(self, len_array):
        if self.o.use_gpu_param:
            num = len(len_array)
            self.all_trans_param = cpfzeros((num, self.o.max_len + 1, self.o.max_len + 1, self.o.cv))
            self.all_dec_param = cpfzeros((num, self.o.max_len + 1, 2, 2, 2))
        else:
            self.all_dec_param = []
            self.all_trans_param = []
            for length in len_array:
                self.all_trans_param.append(npfzeros((length, length, self.o.cv)))
                self.all_dec_param.append(npfzeros((length, 2, 2, 2)))

    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, 'dmv'))

    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, 'dmv')))

    @staticmethod
    def input_gaurd(tag_array):
        """ add ROOT node at position 0 """
        batch_size = tag_array.shape[0]
        tag_array = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device='cuda'), tag_array], axis=1)
        return tag_array
