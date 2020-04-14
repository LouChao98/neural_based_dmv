import os
from dataclasses import dataclass

from module.dmv_torch import DMV, DMVOptions, DMVProb
from utils.common import *
from utils.data import ConllDataset, WSJ_POS
from utils.functions import calculate_uas, print_to_file
from utils.runner import Model, Runner, RunnerOptions, Logger


@dataclass
class DMVModelOptions(DMVOptions, RunnerOptions):
    save_parse_result: bool = False

    # overwrite default opotions
    train_ds: str = 'data/wsj10_tr'
    dev_ds: str = 'data/wsj10_d'
    test_ds: str = 'data/wsj10_te'

    batch_size: int = 64
    max_epoch: int = 200
    early_stop: int = 20
    compare_field: str = 'likelihood'
    save_best: bool = True
    show_log: bool = True

    run_dev: bool = True
    run_test: bool = True

    e_step_mode: str = 'em'
    cv: int = 2
    count_smoothing: float = 0.01
    param_smoothing: float = 0.01

    lr: float = 0.1


PRETRAIN = 4


class DMVModel(Model):
    def __init__(self, o: DMVModelOptions, r: Runner):
        """pre init, DO NOT access any value of o. """
        self.o = o
        self.r = r

    def build(self):
        """init with self.o"""
        self.dmv = DMV(self.o).cuda()

    def train_init(self, epoch_id, dataset):
        self.r.evaluate('test')
        dataset.build_batchs(self.o.batch_size, False, True)

    def train_one_step(self, epoch_id, batch_id, one_batch):
        if len(one_batch[0]) < self.o.batch_size:
            return {'likelihood': 0}
        self.dmv.zero_grad()
        pos_array = torch.tensor(one_batch[1], dtype=torch.long, device='cuda')
        pos_array = self.dmv.prepare_tag_array(pos_array)
        len_array = one_batch[3]

        scores = self.dmv.prepare_scores(pos_array, True)
        ll = self.dmv(*scores, len_array)
        ll = torch.sum(ll)

        (-ll / len(len_array)).backward()
        self.dmv.optimizer.step()

        return {'likelihood': ll.item()}

    def train_callback(self, epoch_id, dataset, result):
        """call after each epoch"""
        return {'likelihood': sum(result['likelihood'])}

    def eval_init(self, mode, dataset):
        if mode == 'test' and self.o.save_best and self.r.best is not None:
            self.load(self.r.best_path)

    def eval_one_step(self, mode, batch_id, one_batch):
        pos_array = torch.tensor(one_batch[1], dtype=torch.long, device='cuda')
        pos_array = self.dmv.prepare_tag_array(pos_array)
        len_array = one_batch[3]

        #
        _trans = self.dmv.trans_param.data.clone()
        _dec = self.dmv.dec_param.data.clone()
        _root = self.dmv.root_param.data.clone()

        #
        with torch.no_grad():
            self.dmv.trans_param.data = torch.log(torch.exp(self.dmv.trans_param.data) + self.o.param_smoothing)
            self.dmv.dec_param.data = torch.log(torch.exp(self.dmv.dec_param.data) + self.o.param_smoothing)
            self.dmv.root_param.data = torch.log(torch.exp(self.dmv.root_param.data) + self.o.param_smoothing)
        self.dmv.normalize_param()

        scores = self.dmv.prepare_scores(pos_array)
        out, ll = self.dmv.parse(*scores, len_array)
        out = {k: v for k, v in zip(one_batch[0], out)}
        out['likelihood'] = ll

        #
        self.dmv.trans_param.data = _trans
        self.dmv.dec_param.data = _dec
        self.dmv.root_param.data = _root

        return out

    def eval_callback(self, mode, dataset, result):
        ll = sum(result['likelihood'])
        del result['likelihood']
        # calculate uas
        for k in result:
            result[k] = result[k][0]
        acc, _, _ = calculate_uas(result, dataset, no_pre=True)
        if self.o.save_parse_result and mode == 'test':
            print_to_file(result, dataset, os.path.join(self.r.workspace, 'parsed.txt'))
        return {'uas': acc * 100, 'likelihood': ll}

    def init_param(self, dataset):
        dataset.build_batchs(self.o.batch_size, same_len=True)
        self.dmv.init_param(dataset)

        self.r.logger.write('start')
        self.o.param_smoothing = 0.
        self.r.evaluate('test')

        if PRETRAIN > 0:
            dataset.build_batchs(len(dataset))
            one_batch = dataset.batch_data[0]
            pos_array = torch.tensor(one_batch[1], dtype=torch.long, device='cuda')
            pos_array = self.dmv.prepare_tag_array(pos_array)
            trans_scores, dec_scores = self.dmv.prepare_scores(pos_array, False)
            trans_scores.retain_grad()
            dec_scores.retain_grad()

            self.dmv.zero_grad()
            _e_step_mode = self.dmv.o.e_step_mode
            self.dmv.o.e_step_mode = 'viterbi'
            ll = self.dmv(trans_scores, dec_scores, one_batch[3])
            self.dmv.o.e_step_mode = _e_step_mode
            torch.sum(ll).backward()
            trans_count = trans_scores.grad.detach()
            dec_count = dec_scores.grad.detach()

            for _ in range(PRETRAIN):
                dataset.build_batchs(self.o.batch_size, shuffle=True)
                for one_batch in dataset.batch_data:
                    id_array = one_batch[0]
                    sub_trans_count = trans_count[id_array]
                    sub_dec_count = dec_count[id_array]

                    pos_array = torch.tensor(one_batch[1], dtype=torch.long, device='cuda')
                    pos_array = self.dmv.prepare_tag_array(pos_array)
                    trans_scores, dec_scores = self.dmv.prepare_scores(pos_array, True)
                    trans_scores[:, :, 0, :] = 0  # for DEBUG
                    ll = torch.sum(sub_trans_count * trans_scores) + torch.sum(sub_dec_count * dec_scores)
                    self.dmv.zero_grad()
                    (-ll / self.o.batch_size).backward()
                    self.dmv.optimizer.step()

                self.r.logger.write(f'pretrain {_}')
                self.r.evaluate('test')
                # break

    def save(self, folder_path):
        self.dmv.save(folder_path)

    def load(self, folder_path):
        self.dmv.load(folder_path)

    def __str__(self):
        return f'DMV_{self.o.e_step_mode}_{self.o.cv}'


class DMVModelProb(Model):
    def __init__(self, o: DMVModelOptions, r: Runner):
        """pre init, DO NOT access any value of o. """
        self.o = o
        self.r = r

    def build(self):
        """init with self.o"""
        self.dmv = DMVProb(self.o).cuda()

    def train_init(self, epoch_id, dataset):
        # dataset.build_batchs(self.o.batch_size, False, True)
        return

    def train_one_step(self, epoch_id, batch_id, one_batch):
        self.dmv.zero_grad()
        pos_array = torch.tensor(one_batch[1], dtype=torch.long, device='cuda')
        prob = torch.nn.functional.one_hot(pos_array, self.o.num_tag).to(torch.float)
        prob = torch.softmax(prob * 10, dim=2)
        pos_array, prob = self.dmv.prepare_tag_array(pos_array, prob)
        len_array = one_batch[3]

        scores = self.dmv.prepare_scores(prob)
        ll = self.dmv(*scores, prob, len_array)
        ll = torch.sum(ll)
        (-ll).backward()

        if epoch_id < 10:
            self.dmv.update_param_use_count()
        else:
            self.dmv.optimizer.step()
        self.dmv.normalize_param()

        return {'likelihood': ll.item()}

    def train_callback(self, epoch_id, dataset, result):
        """call after each epoch"""
        return {'likelihood': sum(result['likelihood'])}

    def eval_init(self, mode, dataset):
        if mode == 'test' and self.r.best is not None and self.o.save_best:
            self.load(self.r.best_path)

    def eval_one_step(self, mode, batch_id, one_batch):
        pos_array = torch.tensor(one_batch[1], dtype=torch.long, device='cuda')
        prob = torch.nn.functional.one_hot(pos_array, self.o.num_tag).to(torch.float)
        pos_array, prob = self.dmv.prepare_tag_array(pos_array, prob)
        len_array = one_batch[3]

        scores = self.dmv.prepare_scores(prob)
        out, ll = self.dmv.parse(*scores, prob, len_array)
        out = {k: v for k, v in zip(one_batch[0], out)}
        out['likelihood'] = ll
        return out

    def eval_callback(self, mode, dataset, result):
        ll = sum(result['likelihood'])
        del result['likelihood']
        # calculate uas
        for k in result:
            result[k] = result[k][0]
        acc, _, _ = calculate_uas(result, dataset, no_pre=True)
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
    def __init__(self, o: DMVModelOptions):
        m = DMVModel(o, self)
        super().__init__(m, o, Logger())

        print(self.workspace)

    def load(self):
        self.train_ds = ConllDataset(self.o.train_ds, pos_vocab=WSJ_POS)
        self.dev_ds = ConllDataset(self.o.dev_ds, pos_vocab=WSJ_POS)
        self.test_ds = ConllDataset(self.o.test_ds, pos_vocab=WSJ_POS)

        self.dev_ds.build_batchs(10000)
        self.test_ds.build_batchs(10000)

        self.o.max_len = 10
        self.o.num_tag = len(WSJ_POS)


if __name__ == '__main__':
    options = DMVModelOptions()
    options.parse()
    runner = DMVModelRunner(options)
    # runner.evaluate('test')
    # exit(0)
    runner.start()
