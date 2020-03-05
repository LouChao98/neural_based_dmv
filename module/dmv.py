import os
from typing import Union, Tuple
from dataclasses import dataclass

import cupyx as cpx

from numba import njit, prange
from utils.common import *
from utils.options import Options
from module.eisner import constituent_index, batch_inside, batch_outside, batch_parse

SOFTMAX_EM_SIGMA_TYPE = Union[float, Tuple[float, float, int]]


@dataclass
class DMVOptions(Options):
    num_tag: int = 0
    max_len: int = 10  # max len for all dataset
    cv: int = 1  # 1 or 2
    e_step_mode: str = 'em'  # em or viterbi
    count_smoothing: float = 1e-1  # smooth for init_param
    param_smoothing: float = 1e-1  # smooth for m_step
    use_gpu_param: bool = False

    # ===== extentions =====

    # see `Unambiguity Regularization for Unsupervised Learning of Probabilistic Grammars`
    use_softmax_em: bool = False
    # if Tuple[float, float, int], use annealing
    # Tuple[float, float, int] means start_sigma, end_sigma, duration.
    softmax_em_sigma: SOFTMAX_EM_SIGMA_TYPE = (1., 0., 100)
    # if sigma bigger than this threshold, run viterbi to avoid overflow
    softmax_em_sigma_threshold: float = 0.9
    # if True, call softmax_em_step automatically when m_step is called.
    softmax_em_auto_step: bool = True

    # backoff_rate*p(r|child_tag, dir, cv) + (1-backoff_rate)*p(r|parent_tag, child_tag, dir, cv)
    # FIXME bad code
    use_child_backoff: bool = False
    child_backoff_rate: float = 0.33

    # function mask is moved to DMV.set_function_mask


class DMV:
    """Dependency Model with Valence

    See module/eisner.py for inside-outside algorithm.
    See model/dmv.py for whole training or testing pipeline.

    Note: there is randomness when using scatter_add because it uses atomicAdd
    """

    def __init__(self, o: DMVOptions):
        self.o = o
        self.initializing = True  # enable ndmv support if False

        # if self.initializing is False, this param will be used
        # instead of self.trans_param and self.dec_param.
        self.all_trans_param = None
        self.all_dec_param = None

        # root_param: child_pos
        # trans_param: head_pos, child_pos, direction, cv
        # dec_param: head_pos, direction, dv, decision
        self.root_param = cpfzeros(self.o.num_tag)
        self.trans_param = cpfzeros((self.o.num_tag, self.o.num_tag, 2, self.o.cv))
        self.dec_param = cpfzeros((self.o.num_tag, 2, 2, 2))

        self.root_counter = cpfzeros(self.o.num_tag)
        self.batch_trans_trace = None
        self.batch_dec_trace = None

        assert not (o.use_softmax_em and o.e_step_mode == 'viterbi'), "softmax em need e_step_mode=em"
        self.softmax_em_cursor = 0
        self.softmax_em_current_sigma = 0.
        self.softmax_em_step()

        self.function_mask_set = None  # for UD

        self.trans_buffer = None  # pinned memory for faster CPU-GPU transfer
        self.dec_buffer = None

    def parse(self, id_array: np.ndarray, tag_array: cp.ndarray, len_array: np.ndarray):
        tag_array = self.input_gaurd(tag_array)
        trans_scores, dec_scores = self.get_scores(id_array, tag_array, len_array)
        heads, *_ = batch_parse(trans_scores, dec_scores, len_array)

        heads = heads.get()
        parse_results = {id_array[i]: heads[i] for i in range(id_array.shape[0])}
        return parse_results

    def e_step(self, id_array: np.ndarray, tag_array: cp.ndarray, len_array: np.ndarray):
        batch_size, real_len = tag_array.shape

        # sentence_transition_trace:  batch, h, m, dir, cv
        # sentence_decision_trace:    batch, h, m, dir, dv, decision.
        #   although we have h==m for stop and h!=m for go, we keep
        #   decision-axis for simplify (as well as direction-axis).
        self.batch_trans_trace = cpfzeros((batch_size, real_len, real_len, 2, self.o.cv))
        self.batch_dec_trace = cpfzeros((batch_size, real_len, real_len, 2, 2, 2))

        tag_array = self.input_gaurd(tag_array)

        if self.o.e_step_mode == 'viterbi' or (
                self.o.use_softmax_em and self.softmax_em_current_sigma >= self.o.softmax_em_sigma_threshold):
            return self.run_viterbi_estep(id_array, tag_array, len_array)
        if self.o.e_step_mode == 'em':
            return self.run_em_estep(id_array, tag_array, len_array)

        raise NotImplementedError

    def run_em_estep(self, id_array, tag_array, len_array):
        batch_size, fake_len = tag_array.shape

        trans_scores, dec_scores = self.get_scores(id_array, tag_array, len_array)
        if self.o.use_softmax_em:
            trans_scores *= 1 / (1 - self.softmax_em_current_sigma)
            dec_scores *= 1 / (1 - self.softmax_em_current_sigma)
        if self.function_mask_set is not None:
            self.function_mask(trans_scores, tag_array)

        if DEBUG:
            assert not cp.isnan(trans_scores).any()
            assert not cp.isnan(dec_scores).any()

        ictable, iitable, prob = batch_inside(trans_scores, dec_scores, len_array)
        octable, oitable = batch_outside(ictable, iitable, trans_scores, dec_scores, len_array)

        if DEBUG:
            assert not cp.isnan(ictable).any()
            assert not cp.isnan(iitable).any()
            assert not cp.isnan(octable).any()
            assert not cp.isnan(oitable).any()

        span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = constituent_index(fake_len)
        len_array = cpasarray(len_array)
        prob = cp.expand_dims(prob, 1)

        for h in range(fake_len):
            for m in range(1, fake_len):
                len_mask = ((h > len_array) | (m > len_array))

                if h == m:
                    for direction in range(2):
                        span_id = span2id[h, h, direction]
                        count = cp.exp(ictable[:, span_id, :] + octable[:, span_id, :] - prob)
                        count[len_mask] = 0.
                        self.batch_dec_trace[:, h - 1, m - 1, direction, :, STOP] = count
                else:
                    direction = 0 if h > m else 1
                    span_id = span2id[m, h, direction] if direction == 0 else span2id[h, m, direction]
                    count = cp.exp(iitable[:, span_id, :] + oitable[:, span_id, :] - prob)
                    count[len_mask] = 0.

                    if h == 0:
                        cpx.scatter_add(self.root_counter, tag_array[:, m], cp.sum(count, axis=1))
                    else:
                        self.batch_trans_trace[:, h - 1, m - 1, direction, :] = \
                            cp.sum(count, axis=1, keepdims=True) if self.o.cv == 1 else count
                        self.batch_dec_trace[:, h - 1, m - 1, direction, :, GO] = count

        batch_likelihood = cp.sum(prob).get()
        if self.o.use_softmax_em:
            batch_likelihood *= (1 - self.softmax_em_current_sigma)
        return batch_likelihood

    def run_viterbi_estep(self, id_array, tag_array, len_array):
        batch_size, sentence_len = tag_array.shape

        trans_scores, dec_scores = self.get_scores(id_array, tag_array, len_array)
        if self.function_mask_set is not None:
            self.function_mask(trans_scores, tag_array)

        heads, head_valences, valences = batch_parse(trans_scores, dec_scores, len_array)

        len_array = cpasarray(len_array)
        batch_arange = cp.arange(batch_size)
        batch_likelihood = cpfzeros(1)

        for m in range(1, sentence_len):
            h = heads[:, m]
            direction = (h <= m).astype(cpi)
            h_valence = head_valences[:, m]
            m_valence = valences[:, m]
            m_child_valence = h_valence if self.o.cv > 1 else cp.zeros_like(h_valence)

            len_mask = ((h <= len_array) & (m <= len_array))
            if DEBUG and ((m <= len_array) & (h > len_array)).any():
                print('find bad arc')

            batch_likelihood += cp.sum(dec_scores[batch_arange, m, 0, m_valence[:, 0], STOP][len_mask])
            batch_likelihood += cp.sum(dec_scores[batch_arange, m, 1, m_valence[:, 1], STOP][len_mask])
            self.batch_dec_trace[batch_arange, m - 1, m - 1, 0, m_valence[:, 0], STOP] = len_mask
            self.batch_dec_trace[batch_arange, m - 1, m - 1, 1, m_valence[:, 1], STOP] = len_mask

            head_mask = h == 0
            mask = head_mask * len_mask
            if mask.any():
                # when use_torch_in_cupy_malloc(), mask.any()=False will raise a NullPointer error
                batch_likelihood += cp.sum(trans_scores[:, 0, m, 0][mask])
                cpx.scatter_add(self.root_counter, tag_array[:, m], mask)

            head_mask = ~head_mask
            mask = head_mask * len_mask
            if mask.any():
                batch_likelihood += cp.sum(trans_scores[batch_arange, h, m, m_child_valence][mask])
                batch_likelihood += cp.sum(dec_scores[batch_arange, h, direction, h_valence, GO][mask])
                self.batch_trans_trace[batch_arange, h - 1, m - 1, direction, m_child_valence] = mask
                self.batch_dec_trace[batch_arange, h - 1, m - 1, direction, h_valence, GO] = mask

        return batch_likelihood.get()[0]

    def m_step(self, trans_counter=None, dec_counter=None, update_root=True):
        if update_root:
            assert not DEBUG or not cp.isnan(self.root_counter).any()
            self.root_counter += self.o.param_smoothing
            root_sum = cp.sum(self.root_counter)
            cp.log(self.root_counter / root_sum, out=self.root_param)

        if trans_counter is not None:
            assert not DEBUG or not cp.isnan(trans_counter).any()
            trans_counter += self.o.param_smoothing
            if self.o.use_child_backoff:
                trans_counter = self.apply_child_backoff(trans_counter)
            child_sum = cp.sum(trans_counter, axis=1, keepdims=True)
            cp.log(trans_counter / child_sum, out=self.trans_param)

        if dec_counter is not None:
            assert not DEBUG or not cp.isnan(dec_counter).any()
            dec_counter += self.o.param_smoothing
            decision_sum = cp.sum(dec_counter, axis=3, keepdims=True)
            cp.log(dec_counter / decision_sum, out=self.dec_param)

        if self.o.use_softmax_em and self.o.softmax_em_auto_step:
            self.softmax_em_step()

    def init_param(self, dataset, getter=None):
        # require same_len
        harmonic_sum = [0., 1.]

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
                        cpx.scatter_add(self.dec_param, (pos, direction, HASCHILD, GO), change[i, direction])
                        cpx.scatter_add(norm_counter, (pos, direction, NOCHILD, STOP), -1)
                        cpx.scatter_add(norm_counter, (pos, direction, HASCHILD, STOP), 1)
                        cpx.scatter_add(self.dec_param, (pos, direction, NOCHILD, STOP), 1)
                    else:
                        cpx.scatter_add(self.dec_param, (pos, direction, NOCHILD, STOP), 1)

        def first_child_update(norm_counter):
            all_param = self.dec_param.flatten()
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
            cpx.scatter_add(self.root_param, pos_array.flatten(), 1. / word_num)
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
                        cpx.scatter_add(self.trans_param, (head_pos, child_pos, direction), diff)
                        change[head_i, direction] += diff
            update_decision(change, norm_counter, pos_array)

        self.trans_param += self.o.count_smoothing
        self.dec_param += self.o.count_smoothing
        self.root_param += self.o.count_smoothing

        es = first_child_update(norm_counter)
        norm_counter *= 0.9 * es
        self.dec_param += norm_counter

        root_param_sum = cp.sum(self.root_param)
        trans_param_sum = cp.sum(self.trans_param, axis=1, keepdims=True)
        decision_param_sum = cp.sum(self.dec_param, axis=3, keepdims=True)

        self.root_param /= root_param_sum
        self.trans_param /= trans_param_sum
        self.dec_param /= decision_param_sum

        cp.log(self.trans_param, out=self.trans_param)
        cp.log(self.root_param, out=self.root_param)
        cp.log(self.dec_param, out=self.dec_param)

    def init_pretrained(self, dataset, getter=None):

        def recovery_one(heads):
            left_most = np.arange(len(heads))
            right_most = np.arange(len(heads))
            for idx, each_head in enumerate(heads[1:]):
                if idx < left_most[each_head]:
                    left_most[each_head] = idx
                if idx > right_most[each_head]:
                    right_most[each_head] = idx

            valences = np.empty((len(heads), 2), dtype=np.int)
            head_valences = np.empty(len(heads), dtype=np.int)

            for idx, each_head in enumerate(heads):
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

        for idx, instance in enumerate(dataset.instance):
            one_heads = npiempty(instance.len + 1)
            one_heads[0] = -1
            one_heads[1:] = instance.parent_id
            one_valences, one_head_valences = recovery_one(one_heads)
            heads[idx] = heads
            valences[idx] = one_valences
            head_valences = one_head_valences
            print('checkpoint')

    def apply_child_backoff(self, transition_counter):
        """backoffs      [1, child_pos, directoin, cv]"""
        backoffs = cp.sum(transition_counter, axis=0, keepdims=True)
        backoff_norms = cp.sum(backoffs, axis=1, keepdims=True)

        equal_prob = 1. / self.o.num_tag

        bad_mask = backoff_norms <= 1e-6
        backoffs[cp.tile(bad_mask, (1, self.o.num_tag, 1, 1))] = equal_prob
        backoff_norms[bad_mask] = 1.
        backoffs /= backoff_norms

        transition_counter_norms = cp.sum(transition_counter, axis=1, keepdims=True)
        transition_counter /= transition_counter_norms
        transition_counter[cp.isnan(transition_counter)] = equal_prob

        return (1 - self.o.child_backoff_rate) * transition_counter + self.o.child_backoff_rate * backoffs

    def softmax_em_step(self):
        if isinstance(self.o.softmax_em_sigma, float):
            self.softmax_em_current_sigma = self.o.softmax_em_sigma
        else:
            start, end, duration = self.o.softmax_em_sigma
            speed = (end - start) / duration
            self.softmax_em_current_sigma = max(
                end, start + speed * self.softmax_em_cursor)
            self.softmax_em_cursor += 1

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
        # scores:           batch, head, child, cv
        # decision_scores:  batch, position, direction, dv, decision
        batch_size, sentence_len = tag_array.shape

        trans_param = cp.expand_dims(self.trans_param, 0)
        head_pos_index = tag_array.reshape(*tag_array.shape, 1, 1, 1)
        child_pos_index = tag_array.reshape(batch_size, 1, sentence_len, 1, 1)
        scores = cp.take_along_axis(cp.take_along_axis(trans_param, head_pos_index, 1), child_pos_index, 2)
        index = (1 - cp.tri(sentence_len, k=-1, dtype=cpi)).reshape(1, sentence_len, sentence_len, 1, 1)
        scores = cp.take_along_axis(scores, index, 3).squeeze(3)

        decision_param = cp.expand_dims(self.dec_param, 0)
        head_pos_index = tag_array.reshape(*tag_array.shape, 1, 1, 1)
        decision_scores = cp.take_along_axis(decision_param, head_pos_index, 1)
        decision_scores[:, 0] = 0

        root_param = cp.expand_dims(self.root_param, 0)
        root_scores = cp.expand_dims(cp.take_along_axis(root_param, tag_array, 1), -1)
        scores[:, 0, :, :] = root_scores
        scores[:, :, 0, :] = -cp.inf
        return scores, decision_scores

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
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'transition_param.npy'), cp.asnumpy(self.trans_param))
        np.save(os.path.join(path, 'root_param.npy'), cp.asnumpy(self.root_param))
        np.save(os.path.join(path, 'decision_param.npy'), cp.asnumpy(self.dec_param))

    def load(self, path):
        self.trans_param[:] = cpasarray(np.load(os.path.join(path, 'transition_param.npy')))
        self.root_param[:] = cpasarray(np.load(os.path.join(path, 'root_param.npy')))
        self.dec_param[:] = cpasarray(np.load(os.path.join(path, 'decision_param.npy')))

    @staticmethod
    def input_gaurd(tag_array):
        """ add ROOT node at position 0 """
        batch_size = tag_array.shape[0]
        tag_array = cp.concatenate([cpizeros((batch_size, 1)), tag_array], axis=1)
        return tag_array
