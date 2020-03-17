import os
from dataclasses import dataclass
from utils.common import *
from utils.data import ConllDataset, WSJ_POS
from utils.runner import Model, Runner, RunnerOptions, Logger
from utils.functions import calculate_uas, print_to_file
from module.dmv import DMV, DMVOptions


@dataclass
class DMVModelOptions(DMVOptions, RunnerOptions):
    save_parse_result: bool = False

    # overwrite default opotions
    train_ds: str = 'data/wsj10_tr'
    dev_ds: str = 'data/wsj10_d'
    test_ds: str = 'data/wsj10_te'

    batch_size: int = 10000
    max_epoch: int = 200
    early_stop: int = 20
    compare_field: str = 'likelihood'
    save_best: bool = True
    show_log: bool = True

    run_dev: bool = True
    run_test: bool = True

    e_step_mode: str = 'em'
    cv: int = 2
    count_smoothing: float = 0.1
    param_smoothing: float = 0.1

    use_softmax_em: bool = True
    softmax_em_sigma: tuple = (1, 0, 100)
    softmax_em_sigma_threshold: float = 0.8
    softmax_em_auto_step: bool = True


class DMVModel(Model):
    def __init__(self, o: DMVModelOptions, r: Runner):
        """pre init, DO NOT access any value of o. """
        self.o = o
        self.r = r

    def build(self):
        """init with self.o"""
        self.trans_counter = cpfempty((self.o.num_tag, self.o.num_tag, 2, self.o.cv))
        self.dec_counter = cpfempty((self.o.num_tag, 2, 2, 2))
        self.dmv = DMV(self.o)

    def train_init(self, epoch_id, dataset):
        """call before each epoch"""
        self.trans_counter.fill(0.)
        self.dec_counter.fill(0.)
        self.dmv.reset_root_counter()

    def train_one_step(self, epoch_id, batch_id, one_batch):
        id_array = one_batch[0]
        pos_array = cpasarray(one_batch[1])
        len_array = one_batch[3]

        ll = self.dmv.e_step(id_array, pos_array, len_array)
        d, t = self.dmv.get_batch_counter_by_tag(pos_array, mode=1)
        self.trans_counter += t[0]
        self.dec_counter += d[0]
        return {'likelihood': ll}

    def train_callback(self, epoch_id, dataset, result):
        """call after each epoch"""
        self.dmv.m_step(self.trans_counter, self.dec_counter, True)
        return {'likelihood': sum(result['likelihood'])}

    def eval_init(self, mode, dataset):
        if mode == 'test' and self.r.best is not None and self.o.save_best:
            self.load(self.r.best_path)

    def eval_one_step(self, mode, batch_id, one_batch):
        id_array = one_batch[0]
        pos_array = cpasarray(one_batch[1])
        len_array = one_batch[3]

        out = self.dmv.parse(id_array, pos_array, len_array)
        out['likelihood'] = self.dmv.e_step(id_array, pos_array, len_array)
        return out

    def eval_callback(self, mode, dataset, result):
        ll = sum(result['likelihood'])
        del result['likelihood']
        # calculate uas
        for k in result:
            result[k] = result[k][0]
        acc, _, _ = calculate_uas(result, dataset)
        if self.o.save_parse_result and mode == 'test':
            print_to_file(result, dataset, os.path.join(self.r.workspace, 'parsed.txt'))
        return {'uas': acc * 100, 'likelihood': ll}

    def init_param(self, dataset):
        dataset.build_batchs(self.o.batch_size, same_len=True)
        self.dmv.init_param(dataset)
        dataset.build_batchs(self.o.batch_size)

    def save(self, folder_path):
        self.dmv.save(folder_path)

    def load(self, folder_path):
        self.dmv.load(folder_path)

    def __str__(self):
        return f'DMV_{self.o.e_step_mode}_{self.o.cv}'


class DMVModelRunner(Runner):
    def __init__(self, o):
        if o.use_softmax_em:
            from utils import common
            common.cpf = cp.float64

        m = DMVModel(o, self)
        super().__init__(m, o, Logger())

    def load(self):
        self.train_ds = ConllDataset(self.o.train_ds, pos_vocab=WSJ_POS)
        self.dev_ds = ConllDataset(self.o.dev_ds, pos_vocab=WSJ_POS)
        self.test_ds = ConllDataset(self.o.test_ds, pos_vocab=WSJ_POS)

        self.dev_ds.build_batchs(self.o.batch_size)
        self.test_ds.build_batchs(self.o.batch_size)

        self.o.max_len = 10
        self.o.num_tag = len(WSJ_POS)


if __name__ == '__main__':
    options = DMVModelOptions()
    options.parse()
    runner = DMVModelRunner(options)
    runner.start()
