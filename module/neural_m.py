import os
import typing
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import make_mask


@dataclass
class NeuralMOptions:
    # MStepModel
    num_lex: int = 0  # in dmv, no need to distinguish lex and pos for l-ndmv
    num_lan: int = 1

    dim_pos_emb: int = 10
    dim_word_emb: int = 200
    dim_valence_emb: int = 10
    dim_direction_emb: int = 1
    dim_lan_emb: int = 5

    dim_hidden: int = 15
    dim_pre_out_decision: int = 5
    dim_pre_out_child: int = 12

    dropout: float = 0.2
    optimizer: str = "adam"
    lr: float = 0.01
    lr_decay_rate: float = 1.
    min_lr: float = 0.01

    encoder_mode: str = 'empty'
    encoder_lstm_dim_hidden: int = 200
    encoder_lstm_num_layers: int = 2
    encoder_lstm_dropout: float = 0.2

    use_pos_emb: bool = False
    use_word_emb: bool = False
    use_valence_emb: bool = False
    share_valence_emb: bool = False
    use_direction_emb: bool = False  # if False, use diff linear for diff directions
    use_sentence_emb: bool = False
    use_lan_emb: bool = False

    freeze_word_emb: bool = False
    freeze_pos_emb: bool = False

    # if True, child_out_linear`s weight will be binded with POS emb and WORD emb(if available).
    #   described in NDMV, L-NDMV.
    # if False, child_out_linear`s weight will be randomness
    use_emb_as_w: bool = False
    use_child_pos_emb: bool = False


class Encoder(nn.Module):
    def __init__(self, emb: nn.Embedding, is_pretrained_emb=False):
        super().__init__()
        self.emb = emb
        self.net = None
        self.reset_func = None
        self.out_dim = 0
        self.is_pretrained_emb = is_pretrained_emb
        if self.is_pretrained_emb and emb.weight.requires_grad:
            self._saved_emb = self.emb.weight.data.clone()
        else:
            self._saved_emb = None

    def forward(self, word_array, len_array):
        h = self.emb(word_array)
        h = self.net(h, len_array)
        return h

    def build_empty_encoder(self):
        def encode(h, len_array):
            return h

        def reset():
            return

        self.net = encode
        self.reset_func = reset
        self.out_dim = self.emb.weight.shape[1]

    def build_lstm_encoder(self, hidden_dim, num_layers, dropout):
        net = nn.LSTM(self.emb.weight.shape[1], hidden_dim // 2,
                      num_layers, dropout=dropout, bidirectional=True)
        self.add_module('lstm_encoder', net)

        # noinspection PyTypeChecker
        def encode(h, len_array):
            batch_size, max_len, *_ = h.shape
            h = h.transpose(1, 0).contiguous()
            h = nn.utils.rnn.pack_padded_sequence(h, len_array, enforce_sorted=False)
            h = net(h)[0]
            h = nn.utils.rnn.pad_packed_sequence(h)[0]
            h = h.transpose(1, 0).contiguous()
            return h

        def reset():
            net.reset_parameters()

        self.net = encode
        self.reset_func = reset
        self.out_dim = hidden_dim

    def build_onetime_encoder(self, out_dim):
        self.to_fetch = None

        def encode(h, len_array):
            assert self.to_fetch is not None
            t = self.to_fetch
            self.to_fetch = None
            return t

        def reset(self):
            return
        self.net = encode
        self.reset_func = reset
        self.out_dim = out_dim

    def reset(self):
        assert self.reset_func is not None, 'reset is undefined'
        self.reset_func()
        if self._saved_emb is not None:
            self.emb.weight.data[:] = self._saved_emb
        else:
            self.emb.reset_parameters()

    def __call__(self, *args, **kwargs) -> typing.Any:
        return super().__call__(*args, **kwargs)


class NeuralM(nn.Module):
    def __init__(self, o: NeuralMOptions, word_emb=None, out_pos_emb=None, pos_emb=None):
        super().__init__()
        self.o = o
        self.cv = o.cv
        self.emb_dim = 0

        num_pos = o.num_tag - o.num_lex
        assert o.use_word_emb or o.use_pos_emb

        if o.use_word_emb:
            if word_emb is not None:
                word_emb = nn.Embedding.from_pretrained(torch.tensor(
                    word_emb, dtype=torch.float), freeze=o.freeze_word_emb)
            else:
                word_emb = nn.Embedding(self.o.num_lex + 2, o.dim_word_emb)
            self.word_encoder = Encoder(word_emb, word_emb is not None)

            if o.encoder_mode == 'lstm':
                self.word_encoder.build_lstm_encoder(o.encoder_lstm_dim_hidden, o.encoder_lstm_num_layers,
                                                     o.encoder_lstm_dropout)
            elif o.encoder_mode == 'empty':
                self.word_encoder.build_empty_encoder()
            elif o.encoder_mode == 'onetime':
                self.word_encoder.build_onetime_encoder(self.o.dim_word_emb)
            else:
                raise ValueError("the encoder only supports (lstm, empty, onetime)")

            self.emb_dim += self.word_encoder.out_dim

        if o.use_sentence_emb:
            assert o.use_word_emb, 'if use sentence emb, must use word emb'
            self.emb_dim += self.word_encoder.out_dim

        if o.use_pos_emb:
            if pos_emb is not None:
                pos_emb = nn.Embedding.from_pretrained(torch.tensor(
                    pos_emb, dtype=torch.float), freeze=o.freeze_pos_emb)
            else:
                pos_emb = nn.Embedding(num_pos, o.dim_pos_emb)
            self.pos_encoder: Encoder = Encoder(pos_emb, pos_emb is not None)
            self.pos_encoder.build_empty_encoder()  # TODO
            self.emb_dim += self.pos_encoder.out_dim

        if o.use_lan_emb:
            assert o.num_lan > 1, 'meaningless option'
            self.lan_emb = nn.Embedding(o.num_lan, o.dim_lan_emb)
            self.emb_dim += o.dim_lan_emb

        if o.use_valence_emb:
            if o.share_valence_emb and self.cv == 2:
                self.cv_emb = nn.Embedding(2, o.dim_valence_emb)
                self.dv_emb = self.cv_emb
            else:
                if o.share_valence_emb:
                    print('share_valence_emb reset to False because need the same num of valence')
                self.cv_emb = nn.Embedding(self.cv, o.dim_valence_emb)
                self.dv_emb = nn.Embedding(2, o.dim_valence_emb)
            self.emb_dim += o.dim_valence_emb

        # must be the last emb because here will init nn.
        if o.use_direction_emb:
            self.direction_emb = nn.Embedding(2, self.direction_dim)
            self.emb_dim += self.direction_dim
            self.emb_linear = nn.Linear(self.emb_dim, o.dim_hidden)
            self.left_right_linear = None
        else:
            self.direction_emb, self.emb_linear = None, None
            self.left_right_linear = nn.Linear(self.emb_dim, 2 * o.dim_hidden)

        self.decision_linear = nn.Linear(o.dim_hidden, o.dim_pre_out_decision)
        self.decision_out_linear = nn.Linear(o.dim_pre_out_decision, 2)

        if o.use_emb_as_w:
            w_dim = 0
            if o.use_word_emb:
                w_dim += o.dim_word_emb
            if o.use_pos_emb:
                w_dim += o.dim_pos_emb

            self.child_linear = nn.Linear(o.dim_hidden, o.dim_pre_out_child)

            # for child_out_linear
            if self.o.dim_pre_out_child != w_dim:
                print(f"overwrite o.dim_pre_out_child to {w_dim} because o.use_emb_as_w = True")
            o.dim_pre_out_child = w_dim

            if out_pos_emb is not None:
                self.pos_emb_out = nn.Parameter(torch.tensor(out_pos_emb, dtype=torch.float))
            else:
                self.pos_emb_out = nn.Parameter(torch.empty(num_pos, o.dim_pre_out_child))
                nn.init.normal_(self.pos_emb_out.data)

        else:
            self.child_linear = nn.Linear(o.dim_hidden, o.dim_pre_out_child)
            self.child_out_linear = nn.Linear(o.dim_pre_out_child, o.num_tag)

        self.activate = F.relu
        self.dropout = nn.Dropout(o.dropout)

        self.optimizer_name = o.optimizer
        self.lr = o.lr
        self.lr_decay = o.lr_decay_rate
        if o.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=o.lr)
        elif o.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=o.lr)
        else:
            self.optimizer = None

    def forward(self, arrays, tag_array, traces=None):
        """
        :param arrays:
          a dict which contains arrays. 'id' and 'len' is necessary.
          There are a series of use_X_emb in options to control the usage of arrays.
        :param tag_array:
          a array indicate word`s tag used in dmv, if num_lex=0,
          tag_array should be the same as pos_array .
        :param traces:
          a dict which contains traces.
          If given, forward in 'train' mode, or forward in 'predict' mode
        :return:
        """
        len_array = arrays['len']
        batch_size = len(len_array)
        max_len = arrays['pos'].shape[1]

        to_expand = []
        if self.o.use_pos_emb:
            to_expand.append(self.pos_encoder(arrays['pos'], len_array))
        if self.o.use_word_emb:
            encoded_word = self.word_encoder(arrays['word'], len_array)
            to_expand.append(encoded_word)

            if self.o.use_sentence_emb:
                direction_splitted = encoded_word.view(batch_size, max_len, 2, -1)
                sentence = torch.cat([direction_splitted[torch.arange(batch_size), len_array - 1, 0, :],
                                      direction_splitted[:, 0, 1, :]], -1)
                to_expand.append(sentence)

        len_mask = make_mask(len_array, max_len)

        if traces is None:
            expanded, d, v, _ = self.prepare_decision(to_expand, len_mask, batch_size, max_len)
            decision_param = self.real_forward(expanded[:], d, v, True).reshape(batch_size, max_len, 2, 2, 2)

            if self.cv != 2:
                expanded, d, v, _ = self.prepare_trainsition(to_expand, len_mask, batch_size, max_len)
            h = self.real_forward(expanded, d, v, False)
            transition_param = self.transition_param_helper(tag_array, h, valid_direction=True)

            return decision_param, transition_param

        loss = torch.tensor(0., device='cuda')

        decision_trace = traces['decision']
        decision_trace = decision_trace.view(-1)

        expanded, d, v, mask = self.prepare_decision(to_expand, len_mask, batch_size, max_len)
        h = self.real_forward(expanded[:], d, v, True).view(-1)
        loss += self.loss(h, decision_trace, mask)  # & decision_mask.view(-1))

        transition_trace = traces['transition']
        transition_trace = transition_trace.view(-1)

        if self.cv != 2:
            expanded, d, v, mask = self.prepare_trainsition(to_expand, len_mask, batch_size, max_len)
        else:
            *_, mask = self.prepare_trainsition(to_expand, len_mask, batch_size, max_len, only_mask=True)
        h = self.real_forward(expanded, d, v, False)
        h = self.transition_param_helper(tag_array, h, valid_direction=False).view(-1)
        loss += self.loss(h, transition_trace, mask)  # & trace_mask.view(-1))
        return loss

    def real_forward(self, emb_buffer, direction, valence, is_decision):
        emb_buffer.append(self.dv_emb(valence)
                          if is_decision else self.cv_emb(valence))
        if self.o.use_direction_emb:
            emb_buffer.append(self.direction_emb(direction))
        h = torch.cat(emb_buffer, dim=-1)
        del emb_buffer

        h = self.dropout(h)
        if self.o.use_direction_emb:
            h = self.activate(self.emb_linear(h))
        else:
            left_right_h = self.activate(self.left_right_linear(h))
            left_h = left_right_h[:, :self.o.dim_hidden]
            right_h = left_right_h[:, self.o.dim_hidden:]
            left_h[direction == 1, :] = 0.
            right_h[direction == 0, :] = 0.
            h = left_h + right_h
            del left_h, right_h, left_right_h

        h = self.dropout(h)
        if is_decision:
            h = self.decision_out_linear(
                self.activate(self.decision_linear(h)))
        else:
            if self.o.use_emb_as_w:
                w = []
                if self.o.use_word_emb:
                    w.append(torch.cat([self.pos_emb_out, self.word_encoder.emb(self.word_idx)]))
                if self.o.use_pos_emb:
                    all_pos = torch.arange(self.pos_encoder.emb.num_embeddings, device='cuda')
                    w.append(torch.cat([self.pos_encoder.emb(all_pos), self.pos_encoder.emb(self.pos_idx)]))
                w = torch.cat(w, dim=1)
                w = w.T
                h = torch.mm(self.activate(self.child_linear(h)), w)
            else:
                h = self.child_out_linear(self.activate(self.child_linear(h)))
        return torch.log_softmax(h, dim=-1)

    @staticmethod
    def loss(forward_out, target_count, mask):
        batch_loss = -torch.sum(target_count * forward_out * mask) / torch.sum(mask)
        return batch_loss

    @staticmethod
    def prepare_decision(arrays_to_expand, mask, batch_size, max_len):
        # arrays in arrays_to_expand should has shape (batch_size, hidden) or (batch_size, max_len, hidden)
        expanded = []
        for array in arrays_to_expand:
            array = array.view(batch_size, -1, 1, array.shape[-1])
            array = array.expand(-1, max_len, 4, -1)
            array = array.reshape(batch_size * max_len * 4, -1)
            expanded.append(array)

        direction_array = torch.zeros(
            batch_size * max_len, 2, 2, dtype=torch.long, device='cuda')
        direction_array[:, 1, :] = 1
        direction_array = direction_array.view(-1)

        valence_array = torch.zeros(
            batch_size * max_len * 2, 2, dtype=torch.long, device='cuda')
        valence_array[:, 1] = 1
        valence_array = valence_array.view(-1)

        mask = mask.unsqueeze(-1).expand(-1, -1, 8).reshape(-1)
        return expanded, direction_array, valence_array, mask

    def prepare_trainsition(self, arrays_to_expand, mask, batch_size, max_len, only_mask=False):
        if only_mask is False:
            expanded = []
            for array in arrays_to_expand:
                array = array.view(batch_size, -1, 1, array.shape[-1])
                array = array.expand(-1, max_len, 2 * self.cv, -1)
                array = array.reshape(batch_size * max_len * 2 * self.cv, -1)
                expanded.append(array)

            direction_array = torch.zeros(
                batch_size * max_len, 2, self.cv, dtype=torch.long, device='cuda')
            direction_array[:, 1, :] = 1
            direction_array = direction_array.view(-1)

            if self.cv == 2:
                valence_array = torch.zeros(
                    batch_size * max_len * 2, self.cv, dtype=torch.long, device='cuda')
                valence_array[:, 1] = 1
                valence_array = valence_array.view(-1)
            else:
                valence_array = torch.zeros(
                    batch_size * max_len * 2, dtype=torch.long, device='cuda')
        else:
            expanded, direction_array, valence_array = None, None, None

        mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(
            3).expand(-1, -1, -1, 2 * self.cv).reshape(-1)
        return expanded, direction_array, valence_array, mask

    def transition_param_helper(self, group_ids, forward_output, valid_direction=False):
        """convert (batch, seq_len, 2, self.cv, num_tag) to (batch, seq_len, seq_len, [direction,] self.cv)"""
        batch_size, max_len = group_ids.shape
        forward_output = forward_output.view(batch_size, max_len, 2, self.cv, self.o.num_tag)
        index = group_ids.view(batch_size, 1, 1, 1,
                               max_len).expand(-1, max_len, 2, self.cv, -1)
        h = torch.gather(forward_output, 4, index).permute(0, 1, 4, 2, 3).contiguous()
        if valid_direction:
            index = torch.ones(batch_size, max_len, max_len,
                               1, self.cv, dtype=torch.long, device='cuda')
            for i in range(max_len):
                index[:, i, :i] = 0
            h = torch.gather(h, 3, index).squeeze(3)
        return h

    def set_lex(self, word_idx, pos_idx):
        # the order MUST match the converter
        # (word_idx[0], pos_idx[0])=num_pos ...
        self.word_idx = word_idx
        self.pos_idx = pos_idx

    def reduce_lr_rate(self):
        self.lr *= self.lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_decay

    def reset_lr_rate(self):
        self.lr = self.o.lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr

    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, 'model'))

    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, 'model')))

    def reset(self):
        if self.o.use_word_emb:
            self.word_encoder.reset()
        if self.o.use_pos_emb:
            self.pos_encoder.reset()
        if self.o.use_lan_emb:
            self.lan_emb.reset_parameters()
        if self.o.use_valence_emb:
            self.cv_emb.reset_parameters()
            if self.dv_emb is not self.cv_emb:
                self.dv_emb.reset_parameters()
        if self.o.use_direction_emb:
            self.direction_emb.reset_parameters()
            self.emb_linear.reset_parameters()
        else:
            self.left_right_linear.reset_parameters()

        self.decision_linear.reset_parameters()
        self.decision_out_linear.reset_parameters()

        self.child_linear.reset_parameters()
        if self.o.use_emb_as_w:
            nn.init.normal_(self.pos_emb_out.data)
        else:
            self.child_out_linear.reset_parameters()

        self.lr = self.o.lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.o.lr)

    def __call__(self, *args, **kwargs) -> typing.Any:
        return super().__call__(*args, **kwargs)
