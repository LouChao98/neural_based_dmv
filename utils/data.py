import re
import random
from collections import Counter
from itertools import groupby
from utils.common import *


class Vocab:
    def __init__(self, stoi, itos, freeze: bool, unk: str, pad: str):
        self.freeze = False
        self.stoi = stoi or dict()
        self.itos = itos or list()
        self.unk = self.add_one(unk) if unk else None
        self.pad = self.add_one(pad) if pad else None

        self.freeze = freeze

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.itos[item]
        else:
            v = self.stoi.get(item, self.unk)
            if v == self.unk:
                if not self.freeze:
                    i = self.add_one(item)
                    return self.itos[i]
                if self.unk is None:
                    raise KeyError
            return v

    def __len__(self):
        return len(self.itos)

    def add_one(self, x):
        if x not in self.stoi:
            self.stoi[x] = len(self.itos)
            self.itos.append(x)
        return self.stoi[x]

    @staticmethod
    def filte_counter(c: Counter, threshold: int):
        return [x for x, num in c.items() if num >= threshold]

    @staticmethod
    def from_dict(d, freeze=True, unk=None, pad=None):
        itos = []
        for x, idx in d.items():
            if idx > len(itos):
                itos.extend([None] * (idx - len(itos)))
            itos.append(x)
        vocab = Vocab(d, itos, freeze, unk, pad)
        return vocab

    @staticmethod
    def from_list(a, freeze=True, unk=None, pad=None):
        stoi = {v: k for k, v in enumerate(a)}
        vocab = Vocab(stoi, a, freeze, unk, pad)
        return vocab

    @staticmethod
    def empty_vocab(freeze=False, unk=None, pad=None):
        vocab = Vocab(dict(), list(), freeze, unk, pad)
        return vocab


UD_POS = Vocab.from_list('ADV NOUN ADP NUM SCONJ PROPN DET SYM INTJ PART PRON VERB X AUX CONJ ADJ'.split())
WSJ_POS = Vocab.from_list('RB NNP NN WRB NNS VBN UH JJ VB FW CD NNPS PRP VBD IN DT VBZ VBP '
                          'VBG RP $ WP RBR PRP$ CC JJS MD JJR POS EX TO WDT PDT RBS'.split())
BLLIP_POS = Vocab.from_list('RB NNP NN WRB NNS VBN UH JJ VB FW CD NNPS PRP VBD IN DT VBZ VBP '
                            'VBG RP $ WP RBR PRP$ CC JJS MD JJR POS EX TO WDT PDT RBS WP$'.split())


class ConllEntry:
    numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
    __slots__ = (
        "id", "form", "lemma", "pos", "cpos", "feats", "parent_id", "relation", "deps", "misc", "norm",
        "pred_parent_id", "pred_relation")

    def __init__(self, id, form, lemma, pos, cpos, feats, parent_id,
                 relation='-', deps='-', msic='-'):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.cpos = cpos
        self.feats = feats
        self.parent_id = parent_id
        self.relation = relation
        self.deps = deps
        self.misc = msic

        self.norm = self.normalize(self.form)
        self.pred_parent_id = None
        self.pred_relation = None

        self.pos = self.pos.upper()
        self.cpos = self.cpos.upper()

    @staticmethod
    def normalize(word):
        return 'NUM' if ConllEntry.numberRegex.match(word) else word.lower()

    def __repr__(self):
        return f'{self.norm}'

    def __str__(self):
        return f'{self.id}\t{self.form}\t{self.lemma}\t{self.pos}\t{self.cpos}\t{self.feats}\t{self.parent_id}'


class ConllInstance:
    def __init__(self, id, entries, ds=None):
        self.entries = entries
        self.len = len(self.entries)
        self.id = id
        self.ds = ds

        self._pos_np = None
        self._norm_np = None

    def __len__(self):
        return self.len

    def __iter__(self):
        return iter(self.entries)

    def __repr__(self):
        return f'ConllInstance(id={self.id}, len={self.len})'

    def __str__(self):
        return f'ConllInstance(str={" ".join(self.__getattr__("norm"))})'

    def __getattr__(self, item):
        if item == 'pos_np':
            if self._pos_np is None:
                self._pos_np = npasarray(
                    list(map(self.ds.pos_vocab.stoi.__getitem__, self.pos)))
            return self._pos_np
        elif item == 'norm_np':
            if self._norm_np is None:
                assert self.ds.word_vocab is not None
                self._norm_np = npasarray(
                    list(map(self.ds.word_vocab.__getitem__, self.norm)))
            return self._norm_np
        return [e.__getattribute__(item) for e in self.entries]

    def get_raw(self):
        raw_entries = []
        for e in self.entries:
            raw_entries.append(str(e))
        return '\n'.join(raw_entries)

    def remove_entry(self, id):
        """

        :param idx: entry.id
        :return:
        """
        if not (0 < id <= self.len):
            raise IndexError("out of bound")
        for e in self.entries:
            if e.id == id:
                continue
            if e.parent_id == id:
                raise RuntimeError(f"remove {id} will make bad arc")
            if e.parent_id > id:
                e.parent_id -= 1
            if e.id > id:
                e.id -= 1
        self.entries.pop(id - 1)
        self.len -= 1


class ConllDataset:
    def __init__(self, path, word_vocab=None, pos_vocab=None, sort=False,
                 min_len=0, max_len=70):
        self.path = path

        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.sorted = False

        with open(self.path, 'r') as f:
            self.instances = list(filter(lambda i: min_len <= i.len <= max_len,
                                         self.read_instance(f)))
        if sort:
            self.sort_by_len()
        else:
            for id, instance in enumerate(self.instances):
                instance.id = id

        self.batch_data = None

    def __iter__(self):
        return iter(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def __len__(self):
        return len(self.instances)

    def build_word_vocab(self, threshold=1, unk='<UNK>', pad='<PAD>'):
        c = Counter()
        for i in self.instances:
            c.update(i.norm)
        c = Vocab.filte_counter(c, threshold)
        self.word_vocab = Vocab.from_list(c, unk=unk, pad=pad)
        return self.word_vocab

    def build_pos_vocab(self):
        s = set()
        for i in self.instances:
            s.update(i.pos)
        self.pos_vocab = Vocab.from_list(list(s))
        return self.pos_vocab

    def read_instance(self, stream):
        tokens = []
        for line in stream:
            line = line.strip()

            if line == '':
                if tokens:
                    yield ConllInstance(0, tokens, self)
                tokens = []
            elif line[0] == '#':
                continue
            else:
                token = line.split('\t')
                if len(token) <= 10:
                    token[0] = int(token[0])
                    if len(token) >= 7:
                        token[6] = int(token[6]) if token[6] != "_" else -1
                    tokens.append(ConllEntry(*token))
                else:
                    print(f'skip line: "{line}"')
        if tokens:
            yield ConllInstance(0, tokens, self)

    def build_batchs(self, batch_size, same_len=False, shuffle=False):
        """
        same_len, shuffle = T, T: build batchs for each len group and shuffle in groups
        same_len, shuffle = T, F: build batchs for each len group, each group has the same order as self.instances
        same_len, shuffle = F, T: completely shuffle the dataset
        same_len, shuffle = F, F: order is the same as self.instance
        """
        def get_batch_data(data):
            batch_data = []
            len_datas = len(data)
            num_batch = (len_datas + batch_size - 1) // batch_size

            for i in range(num_batch):
                start_idx = i * batch_size
                batch_raw = data[start_idx:start_idx + batch_size]

                id_array = npasarray([i.id for i in batch_raw])
                len_array = npasarray([i.len for i in batch_raw])

                max_len = np.max(len_array)
                pos_array = npizeros((len(batch_raw), max_len))
                for idx, i in enumerate(batch_raw):
                    pos_array[idx, :i.len] = i.pos_np

                if self.word_vocab:
                    word_array = npifull((len(batch_raw), max_len), self.word_vocab.pad)
                    for idx, i in enumerate(batch_raw):
                        word_array[idx, :i.len] = npasarray(
                            list(map(self.word_vocab.__getitem__, i.norm)))
                else:
                    word_array = None

                batch_data.append((id_array, pos_array, word_array, len_array))
            return batch_data

        if same_len:
            if not self.sorted:
                instances = self.instances[:]
                instances.sort(key=lambda i: i.len)
            else:
                instances = self.instances
            groups = [list(g) for _, g in groupby(instances, lambda i: i.len)]
            batched = []
            for group in groups:
                if shuffle:
                    random.shuffle(group)
                batched.extend(get_batch_data(group))
            self.batch_data = batched
        else:
            if shuffle:
                instances = self.instances[:]
                random.shuffle(instances)
            else:
                instances = self.instances
            self.batch_data = get_batch_data(instances)
        return self.batch_data

    def get_batch_iter(self):
        return iter(self.batch_data)

    def get_len(self):
        len_array = npiempty(len(self.instances))
        for i in range(len(self.instances)):
            len_array[i] = self.instances[i].len
        return len_array

    def sort_by_len(self):
        if self.sorted is False:
            self.instances.sort(key=lambda i: i.len)
            for id, instance in enumerate(self.instances):
                instance.id = id
            self.sorted = True

    def shuffle(self):
        self.sorted = False
        random.shuffle(self.instances)
        for id, instance in enumerate(self.instances):
            instance.id = id
