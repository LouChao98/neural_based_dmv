from functools import lru_cache

from utils.common import *

NEGINF = -1e30  # min of float32(which is pytorch`s default) is -3.4e38
D = torch.device('cuda')
ALL = slice(None, None, None)


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


def _sum_op(span_score, mode):
    if mode == 0:  # iitable
        return torch.logsumexp(span_score, dim=0)
    return torch.logsumexp(span_score, dim=(0, 3))


def _max_op(span_score, mode):
    if mode == 0:  # iitable
        return torch.max(span_score, dim=0)
    return torch.max(torch.logsumexp(span_score, dim=3), dim=0)


def get_many_span(table, spans, indexer=None):
    indexer = indexer or ALL
    to_stack = []
    for i in spans:
        assert table[i] is not None
        to_stack.append(table[i][indexer])
    return torch.stack(to_stack)


def sizeof_tensor(t):
    return t.nelement() * t.element_size() / 1024 / 1024


def batch_inside(trans_scores, dec_scores, tag_prop, len_array, mode='sum'):
    # trans_scores:     batch, head, ntag, child, ntag, cv  (head/child means idx in tag seq)
    # dec_scores:       batch, head, ntag, direction, dv, decision
    # tag_prop:         batch, head, ntag
    # len_array:        batch

    op = _sum_op if mode == 'sum' else _max_op

    batch_size, fake_len, ntag, *_, cv = trans_scores.shape
    nspan = (fake_len + 1) * fake_len
    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = constituent_index(fake_len)

    # complete_table:   [nspan], batch, ntag, dv
    # incomplete_table: [nspan], batch, ntag, ntag, dv
    ictable, iitable = [None for _ in range(nspan)], [None for _ in range(nspan)]

    for bs in basic_span:
        ictable[bs] = dec_scores[:, id2span[bs, 0], :, id2span[bs, 2], :, STOP]

    for ij in ijss:
        l, r, direction = id2span[ij]

        # two complete span to form an incomplete span, and add a new arc.
        if direction == 0:
            h, m = r, l
            h_span_id, m_span_id = kjis[ij], ikis[ij]
        else:
            h, m = l, r
            h_span_id, m_span_id = ikis[ij], kjis[ij]

        h_part = get_many_span(ictable, h_span_id, (ALL, ALL, HASCHILD))
        m_part = tag_prop[None, :, m] * get_many_span(ictable, m_span_id, (ALL, ALL, NOCHILD))
        d_part = dec_scores[:, h, :, direction, :, GO]
        t_part = trans_scores[:, h, :, m]

        h_part = h_part.view(-1, batch_size, ntag, 1, 1)
        m_part = m_part.view(-1, batch_size, 1, ntag, 1)
        d_part = d_part.view(1, batch_size, ntag, 1, 2)
        t_part = t_part.view(1, batch_size, ntag, ntag, cv)

        iitable[ij] = op(h_part + m_part + d_part + t_part, mode=0)

        # one complete span and one incomplete span to form an bigger complete span.
        if direction == 0:
            h, m = r, l
            h_span_id, m_span_id = kjcs[ij], ikcs[ij]
        else:
            h, m = l, r
            h_span_id, m_span_id = ikcs[ij], kjcs[ij]

        h_part = get_many_span(iitable, h_span_id)
        m_prob_indexer = torch.tensor([id2span[i, direction] for i in m_span_id], device='cuda', dtype=torch.long)
        m_prob = tag_prop[:, m_prob_indexer].transpose(0, 1).view(-1, batch_size, ntag, 1)
        m_part = m_prob * get_many_span(ictable, m_span_id, (ALL, ALL, NOCHILD, None))
        m_part = m_part.view(-1, batch_size, 1, ntag, 1)
        ictable[ij] = op(h_part + m_part, mode=1)
        pass

    partition_score = [torch.sum(ictable[span2id[0, l, 1]][i, :, NOCHILD]) for i, l in enumerate(len_array)]
    partition_score = torch.stack(partition_score)
    return ictable, iitable, partition_score
