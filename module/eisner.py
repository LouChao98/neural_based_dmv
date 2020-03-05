from functools import lru_cache

from utils.common import *
from utils.functions import (safe_logaddexp as logaddexp,
                             safe_logsumexp as logsumexp,
                             cp_mask_merge as mask_merge)


@lru_cache()
def constituent_index(fake_len):
    """
    helper function to generate span index.
    return np.ndarray and dict
    """
    id2span = []
    span2id = npiempty((fake_len, fake_len, 2))
    for left_idx in range(fake_len):
        for right_idx in range(left_idx, fake_len):
            for direction in range(2):
                span2id[left_idx, right_idx, direction] = len(id2span)
                id2span.append((left_idx, right_idx, direction))
    id2span = npasarray(id2span)

    basic_span = []
    for i in range(fake_len):
        basic_span.append(span2id[i, i, 0])
        basic_span.append(span2id[i, i, 1])
    basic_span = npasarray(basic_span)

    # the order of ijss is important
    ijss = []
    ikis = [[] for _ in range(len(id2span))]
    kjis = [[] for _ in range(len(id2span))]
    ikcs = [[] for _ in range(len(id2span))]
    kjcs = [[] for _ in range(len(id2span))]

    for length in range(1, fake_len):
        for i in range(fake_len - length):
            j = i + length
            ids = span2id[i, j, 0]
            ijss.append(ids)
            for k in range(i, j):
                # two complete spans to form an incomplete span
                ikis[ids].append(span2id[i, k, 1])
                kjis[ids].append(span2id[k + 1, j, 0])
                # one complete span, one incomplete span to form a complete span
                ikcs[ids].append(span2id[i, k, 0])
                kjcs[ids].append(span2id[k, j, 0])
            ids = span2id[i, j, 1]
            ijss.append(ids)
            for k in range(i, j + 1):
                # two complete spans to form an incomplete span
                if k < j and (i != 0 or k == 0):
                    ikis[ids].append(span2id[i, k, 1])
                    kjis[ids].append(span2id[k + 1, j, 0])
                # one incomplete span, one complete span to form a complete span
                if k > i:
                    ikcs[ids].append(span2id[i, k, 1])
                    kjcs[ids].append(span2id[k, j, 1])

    return span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span


@lru_cache(maxsize=3)
def prepare_backtracking(batch_size, fake_len):
    """
    helper function to generate span idx.
    for backtracking
    """
    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = \
        constituent_index(fake_len)

    stack = cpizeros((batch_size, fake_len * 4 - 2, 3))
    id2span = cpasarray(id2span)

    merged_is = []
    merged_is_index = []
    merged_cs = []
    merged_cs_index = []
    for i in range(len(id2span)):
        merged_is_index.append(len(merged_is))
        merged_is.extend(list(zip(ikis[i], kjis[i])))
        merged_cs_index.append(len(merged_cs))
        merged_cs.extend(list(zip(ikcs[i], kjcs[i])))

    merged_is = cpasarray(merged_is)
    merged_is_index = cpasarray(merged_is_index)
    merged_cs = cpasarray(merged_cs)
    merged_cs_index = cpasarray(merged_cs_index)

    heads = cpiempty((batch_size, fake_len))
    head_valences = cpiempty((batch_size, fake_len))
    valences = cpiempty((batch_size, fake_len, 2))

    shape = cpasarray([batch_size, fake_len])

    return heads, head_valences, valences, merged_is, merged_is_index, \
        merged_cs, merged_cs_index, id2span, stack, shape


def batch_inside(trans_scores, dec_scores, len_array):
    # trans_scores: batch, head, child, cv
    # dec_scores: batch, head, direction, dv, decision

    batch_size, fake_len, *_ = trans_scores.shape
    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = \
        constituent_index(fake_len)

    # complete/incomplete_table: batch, all_span, dv
    ictable = cpffull((batch_size, (fake_len + 1) * fake_len, 2), -cp.inf)
    iitable = cpffull((batch_size, (fake_len + 1) * fake_len, 2), -cp.inf)

    iids = id2span[basic_span]
    ictable[:, basic_span, :] = \
        dec_scores[:, iids[:, 0], iids[:, 2], :, STOP].transpose(1, 0, 2)

    for ij in ijss:
        l, r, direction = id2span[ij]

        # two complete span to form an incomplete span, and add a new arc.
        if direction == 0:
            span_inside_i = ictable[:, ikis[ij], NOCHILD, None] \
                + ictable[:, kjis[ij], HASCHILD, None] \
                + trans_scores[:, r, l, None, :] \
                + dec_scores[:, r, direction, None, :, GO]
        else:
            span_inside_i = ictable[:, ikis[ij], HASCHILD, None] \
                + ictable[:, kjis[ij], NOCHILD, None] \
                + trans_scores[:, l, r, None, :]\
                + dec_scores[:, l, direction, None, :, GO]
        iitable[:, ij, :] = logsumexp(span_inside_i, axis=1)

        # one complete span and one incomplete span to form an bigger complete span.
        if direction == 0:
            span_inside_c = ictable[:, ikcs[ij], NOCHILD, None] \
                + iitable[:, kjcs[ij], :]
        else:
            span_inside_c = iitable[:, ikcs[ij], :] \
                + ictable[:, kjcs[ij], NOCHILD, None]
        ictable[:, ij, :] = logsumexp(span_inside_c, axis=1)

    ids = [span2id[0, l, 1] for l in len_array]
    partition_score = ictable[cp.arange(batch_size), ids, NOCHILD]
    return ictable, iitable, partition_score


def batch_outside(ictable, iitable, trans_scores, dec_scores, len_array):
    # trans_scores: batch, head, child, cv
    # dec_scores: batch, head, direction, dv, decision
    batch_size, fake_len, *_ = trans_scores.shape

    # complete/incomplete_table: batch, all_span, dv
    octable = cpffull((batch_size, (fake_len + 1) * fake_len, 2), -cp.inf)
    oitable = cpffull((batch_size, (fake_len + 1) * fake_len, 2), -cp.inf)

    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = \
        constituent_index(fake_len)

    root_id = [span2id[0, l, 1] for l in len_array]
    octable[cp.arange(batch_size), root_id, NOCHILD] = 0.
    len_array = cpasarray(len_array)

    for ij in reversed(ijss):
        l, r, direction = id2span[ij]

        len_mask = (r <= len_array).reshape(batch_size, 1, 1)
        len_mask_inf = cpfzeros((batch_size, 1, 1))
        len_mask_inf[~len_mask] = -cp.inf

        # complete span consists of one incomplete span and one complete span
        if direction == 0:
            outside_ij_cc = octable[:, None, ij, :]
            inside_ik_cc = ictable[:, ikcs[ij], NOCHILD, None]
            inside_kj_ic = iitable[:, kjcs[ij], :]

            outside_ik_cc = mask_merge(outside_ij_cc + inside_kj_ic, len_mask_inf, len_mask)
            outside_kj_ic = mask_merge(outside_ij_cc + inside_ik_cc, len_mask_inf, len_mask)
            outside_ik_cc = logsumexp(outside_ik_cc, axis=2)

            octable[:, ikcs[ij], NOCHILD] = logaddexp(octable[:, ikcs[ij], NOCHILD], outside_ik_cc)
            oitable[:, kjcs[ij], :] = logaddexp(oitable[:, kjcs[ij], :], outside_kj_ic)
        else:
            outside_ij_cc = octable[:, None, ij, :]
            inside_ik_ic = iitable[:, ikcs[ij], :]
            inside_kj_cc = ictable[:, kjcs[ij], NOCHILD, None]

            outside_kj_cc = mask_merge(outside_ij_cc + inside_ik_ic, len_mask_inf, len_mask)
            outside_ik_ic = mask_merge(outside_ij_cc + inside_kj_cc, len_mask_inf, len_mask)
            outside_kj_cc = logsumexp(outside_kj_cc, axis=2)

            octable[:, kjcs[ij], NOCHILD] = logaddexp(octable[:, kjcs[ij], NOCHILD], outside_kj_cc)
            oitable[:, ikcs[ij], :] = logaddexp(oitable[:, ikcs[ij], :], outside_ik_ic)

        # incomplete span consists of two complete spans
        outside_ij_ii = oitable[:, ij, None, :]
        inside_ik_ci = ictable[:, ikis[ij], :]
        inside_kj_ci = ictable[:, kjis[ij], :]

        if direction == 0:
            outside_ik_ci = outside_ij_ii + inside_kj_ci[:, :, None, HASCHILD] \
                + trans_scores[:, None, r, l, :] \
                + dec_scores[:, None, r, direction, :, GO]
            outside_kj_ci = outside_ij_ii + inside_ik_ci[:, :, None, NOCHILD] \
                + trans_scores[:, None, r, l, :] \
                + dec_scores[:, None, r, direction, :, GO]
        else:
            outside_ik_ci = outside_ij_ii + inside_kj_ci[:, :, None, NOCHILD] \
                + trans_scores[:, None, l, r, :] \
                + dec_scores[:, None, l, direction, :, GO]
            outside_kj_ci = outside_ij_ii + inside_ik_ci[:, :, None, HASCHILD] \
                + trans_scores[:, None, l, r, :] \
                + dec_scores[:, None, l, direction, :, GO]

        outside_ik_ci = mask_merge(outside_ik_ci, len_mask_inf, len_mask)
        outside_kj_ci = mask_merge(outside_kj_ci, len_mask_inf, len_mask)

        outside_ik_ci_i = logsumexp(outside_ik_ci, axis=2)
        outside_kj_ci_i = logsumexp(outside_kj_ci, axis=2)

        if direction == 0:
            octable[:, ikis[ij], NOCHILD] = logaddexp(octable[:, ikis[ij], NOCHILD], outside_ik_ci_i)
            octable[:, kjis[ij], HASCHILD] = logaddexp(octable[:, kjis[ij], HASCHILD], outside_kj_ci_i)
        else:
            octable[:, ikis[ij], HASCHILD] = logaddexp(octable[:, ikis[ij], HASCHILD], outside_ik_ci_i)
            octable[:, kjis[ij], NOCHILD] = logaddexp(octable[:, kjis[ij], NOCHILD], outside_kj_ci_i)

    return octable, oitable


def batch_parse(trans_scores, dec_scores, len_array):
    # trans_scores: batch, head, child, cv
    # dec_scores: batch, head, direction, dv, decision

    batch_size, fake_len, *_ = trans_scores.shape
    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = \
        constituent_index(fake_len)

    complete_table = cpffull((batch_size, (fake_len + 1) * fake_len, 2), -cp.inf)
    incomplete_table = cpffull((batch_size, (fake_len + 1) * fake_len, 2), -cp.inf)
    complete_backtrack = cpifull((batch_size, (fake_len + 1) * fake_len, 2), -1)
    incomplete_backtrack = cpifull((batch_size, (fake_len + 1) * fake_len, 2), -1)

    iids = id2span[basic_span]
    complete_table[:, basic_span, :] = \
        dec_scores[:, iids[:, 0], iids[:, 2], :, STOP].transpose(1, 0, 2)

    for ij in ijss:
        l, r, direction = id2span[ij]

        # two complete span to form an incomplete span, and add a new arc.
        if direction == 0:
            span_i = complete_table[:, ikis[ij], NOCHILD, None] \
                + complete_table[:, kjis[ij], HASCHILD, None] \
                + trans_scores[:, r, l, None, :] \
                + dec_scores[:, r, direction, None, :, GO]
        else:
            span_i = complete_table[:, ikis[ij], HASCHILD, None] \
                + complete_table[:, kjis[ij], NOCHILD, None] \
                + trans_scores[:, l, r, None, :] \
                + dec_scores[:, l, direction, None, :, GO]
        incomplete_backtrack[:, ij, :] = cp.argmax(span_i, axis=1)
        incomplete_table[:, ij, :] = cp.take_along_axis(
            span_i, cp.expand_dims(incomplete_backtrack[:, ij, :], 1),
            axis=1).squeeze()

        # one complete span and one incomplete span to form bigger complete span
        if direction == 0:
            span_c = complete_table[:, ikcs[ij], NOCHILD, None]\
                + incomplete_table[:, kjcs[ij], :]
        else:
            span_c = incomplete_table[:, ikcs[ij], :] \
                + complete_table[:, kjcs[ij], NOCHILD, None]
        complete_backtrack[:, ij, :] = cp.argmax(span_c, axis=1)
        complete_table[:, ij, :] = cp.take_along_axis(
            span_c, cp.expand_dims(complete_backtrack[:, ij, :], 1),
            axis=1).squeeze()

    heads, head_valences, valences, merged_is, merged_is_index, \
        merged_cs, merged_cs_index, id2span, stack, shape = \
        prepare_backtracking(batch_size, fake_len)

    root_id = cp.asarray([span2id[0, l, 1] for l in len_array], dtype=cpi)
    backtracking(
        ((batch_size + 255) // 256,), (256,),
        (incomplete_backtrack, complete_backtrack, heads,
            head_valences, valences, merged_is, merged_is_index, merged_cs,
            merged_cs_index, root_id, id2span, stack, shape))
    return heads, head_valences, valences


backtracking = cp.RawKernel("""
#define PUT(_stack, _idx,  _cursor, _a, _b, _c) { \\
    _stack[_idx * stack_ndim0 + _cursor * stack_ndim1 + 0] = _a; \\
    _stack[_idx * stack_ndim0 + _cursor * stack_ndim1 + 1] = _b; \\
    _stack[_idx * stack_ndim0 + _cursor * stack_ndim1 + 2] = _c; \\
    ++_cursor; }

#define POP(_stack, _idx, _cursor, _a, _b, _c) { \\
    --_cursor; \\
    _a = _stack[_idx * stack_ndim0 + _cursor * stack_ndim1 + 0]; \\
    _b = _stack[_idx * stack_ndim0 + _cursor * stack_ndim1 + 1]; \\
    _c = _stack[_idx * stack_ndim0 + _cursor * stack_ndim1 + 2]; }

extern "C" __global__ void
backtracking(
    const int *incomplete_backtrack, const int *complete_backtrack,
    int *head, int *head_valence, int *valence,
    const int *is, const int *is_index, const int *cs, const int *cs_index,
    const int *root_span_id, const int *id2span, int *stack, const int *shape)
{
    // stack[idx] should be (batch_size, max_len * 3, 3) to avoid overflow

    const int n = shape[0];
    const int max_len = shape[1];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const int stack_ndim0 = (max_len * 4 - 2) * 3;
    const int stack_ndim1 = 3;
    const int trace_ndim0 = (max_len + 1) * max_len * 2;
    const int trace_ndim1 = 2;
    const int out_ndim0 = max_len;

    int cursor = 0;
    PUT(stack, idx, cursor, root_span_id[idx], 0, 1)

    while (cursor > 0)
    {
        int span_id, dvalence, complete;
        POP(stack, idx, cursor, span_id, dvalence, complete)

        int l = id2span[span_id * 3 + 0];
        int r = id2span[span_id * 3 + 1];
        int d = id2span[span_id * 3 + 2];

        if (l == r)
            valence[idx * out_ndim0 * 2 + l * 2 + d] = dvalence;
        else
        {
            if (complete == 1)
            {
                int k = complete_backtrack[
                    idx * trace_ndim0 + span_id * trace_ndim1 + dvalence];
                int left_span_id = cs[((cs_index[span_id] + k) << 1) + 0];
                int right_span_id = cs[((cs_index[span_id] + k) << 1) + 1];

                int l_dv = d == 0 ? 0 : dvalence;
                int r_dv = d == 1 ? 0 : dvalence;
                PUT(stack, idx, cursor, left_span_id, l_dv, 1 - d)
                PUT(stack, idx, cursor, right_span_id, r_dv, d)
            }
            else
            {
                int k = incomplete_backtrack[
                    idx * trace_ndim0 + span_id * trace_ndim1 + dvalence];
                int left_span_id = is[((is_index[span_id] + k) << 1) + 0];
                int right_span_id = is[((is_index[span_id] + k) << 1) + 1];

                int h = d == 0 ? r : l;
                int t = d == 1 ? r : l;
                head[idx * out_ndim0 + t] = h;
                head_valence[idx * out_ndim0 + t] = dvalence;
                PUT(stack, idx, cursor, left_span_id, d, 1)
                PUT(stack, idx, cursor, right_span_id, 1 - d, 1)
            }
        }
    }
}
""", 'backtracking')
