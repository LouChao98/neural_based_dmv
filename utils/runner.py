import os
import json
import pickle
import time
import shutil
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from utils.functions import make_sure_dir_exists
from utils.options import Options


class Model:

    def build(self):
        pass

    def train_init(self, epoch_id, dataset):
        pass

    def train_one_step(self, epoch_id, batch_id, one_batch):
        raise NotImplementedError

    def train_callback(self, epoch_id, dataset, result):
        return result

    def eval_init(self, mode, dataset):
        pass

    def eval_one_step(self, mode, batch_id, one_batch):
        raise NotImplementedError

    def eval_callback(self, mode, dataset, result):
        return result

    def init_param(self, dataset):
        pass

    def save(self, folder_path):
        pass

    def load(self, folder_path):
        pass

    def stop_test(self):
        pass

    def __str__(self):
        return 'UntitledModel'


class Logger:
    def __init__(self, o=None):
        self.o = o

        self.train_prefix = '[EPOCH {epoch}]'
        self.dev_prefix = '[DEV]'
        self.test_prefix = '[TEST]'
        self.default_prefix = '[LOG]'

        self._previous = None
        self._f = None

    def _write(self, info):
        if self.o is None:
            print(info)
            return
        if self.o.show_log:
            print(info)
        if self.o.save_log:
            if self._f is None:
                self._f = open(self.o.workspace + '/log.txt', 'w')
            self._f.write(info + '\n')
            self._f.flush()

    def get_prefix(self, prefix=None, train_epoch=-1, is_dev=False, is_test=False):
        if prefix is not None:
            pass
        elif train_epoch >= 0:
            prefix = self.train_prefix.format(epoch=train_epoch)
        elif is_dev:
            prefix = self.dev_prefix
        elif is_test:
            prefix = self.test_prefix
        else:
            prefix = self.default_prefix

        if prefix == self._previous:
            prefix = ' ' * len(prefix)
        else:
            self._previous = prefix
        return prefix

    def write(self, info, *, time=None, prefix=None, train_epoch=-1, is_dev=False, is_test=False):
        prefix = self.get_prefix(prefix, train_epoch, is_dev, is_test)
        suffix = f'({time:.3f}s)' if time else ''
        self._write(' '.join([prefix, info, suffix]))

    def write_result(self, result, *, prefix=None, time=None, train_epoch=-1, is_dev=False, is_test=False):
        prefix = self.get_prefix(prefix, train_epoch, is_dev, is_test)
        result = [f"{k} = {v:{'.2f' if isinstance(v,float) else ''}}" for k, v in result.items()]
        result = ', '.join(result)
        suffix = f'({time:.2f}s)' if time else ''
        self._write(' '.join([prefix, result, suffix]))


@dataclass
class RunnerOptions(Options):
    train_ds: str = ''
    dev_ds: str = ''
    test_ds: str = ''

    batch_size: int = 1

    # train
    max_epoch: int = 1
    early_stop: int = 0  # stop if n epochs no improvement, 0=No early stop

    # eval
    compare_field: str = ''
    run_dev: bool = False
    run_test: bool = False

    # workspace
    workspace: str = ''
    many_runs: bool = True  # use time as sub folder name

    save_checkpoint_freq: int = 0  # save every n epoch, 0=no save
    save_checkpoint_max: int = 5  # max num of checkpoints
    save_best: bool = True
    save_log: bool = True
    show_log: bool = True
    show_best_hit: bool = False

    # recovery
    load_model: str = ''  # path to folder. NOT workspace
    load_train: str = ''  # eg. <path_to_status.json>[!-2]


class Runner:
    def __init__(self, m: Model, o: RunnerOptions, l: Logger):
        self.o = o
        self.model = m
        self.logger = l

        self.train_ds = None
        self.dev_ds = None
        self.test_ds = None
        self.load()
        self.model.build()

        # recovery
        if o.load_train:
            self.load_train(o.load_train)
        elif o.load_model:
            self.load_model(o.load_model)
        else:
            self.model.init_param(self.train_ds)

        # prepare workspace
        if not o.workspace:
            o.workspace = f'./output/{self.model}'
        if o.many_runs:
            o.workspace = o.workspace + f'/{datetime.now()}'
        self.workspace = o.workspace
        make_sure_dir_exists(o.workspace)
        if o.save_best:
            self.best_path = o.workspace + '/best'
            make_sure_dir_exists(self.best_path)
        if o.save_checkpoint_freq:
            self.checkpoints_path = o.workspace + '/checkpoints'
            make_sure_dir_exists(self.checkpoints_path)

        self.best_epoch = -1
        self.best = None
        self.prev_train_result = None
        self.prev_dev_result = None

        self.checkpoints = []

        self.o.save(self.workspace + '/options.json')

    def train(self):
        train_start_time = time.time()
        for epoch in range(self.o.max_epoch):
            start_time = time.time()

            self.model.train_init(epoch, self.train_ds)
            result = defaultdict(list)
            for batch, one_batch in enumerate(self.train_ds.get_batch_iter()):
                one_result = self.model.train_one_step(epoch, batch, one_batch)
                for k, v in one_result.items():
                    result[k].append(v)
            result = self.model.train_callback(epoch, self.train_ds, result)
            self.prev_train_result = result

            end_time = time.time()
            self.logger.write_result(result, time=end_time - start_time, train_epoch=epoch)

            if self.o.run_dev:
                result = self.evaluate('dev')
                self.prev_dev_result = result
                if self.best is None or result[self.o.compare_field] > self.best[self.o.compare_field]:
                    self.best = result
                    self.best_epoch = epoch
                    if self.o.save_best:
                        self.model.save(self.best_path)
                    if self.o.show_best_hit:
                        self.logger.write('Find new best!', is_dev=True)
                if self.o.early_stop and epoch - self.best_epoch >= self.o.early_stop:
                    self.logger.write('Early stop triggered.', is_dev=True)
                    break

            if self.o.save_checkpoint_freq and (epoch + 1) % self.o.save_checkpoint_freq == 0:
                self.save_checkpoint(epoch, )

        if self.best is not None:
            self.logger.write(f'Get the best at epoch {self.best_epoch}.',
                              prefix='[SUMMARY]', time=time.time() - train_start_time)
            self.logger.write_result(self.best, prefix='[SUMMARY]')

        if self.o.run_test:
            self.evaluate('test')

    def evaluate(self, mode):
        start_time = time.time()
        if mode == 'dev':
            ds = self.dev_ds
        elif mode == 'test':
            ds = self.test_ds
        else:
            raise ValueError(f"bad mode: {mode}")

        self.model.eval_init(mode, ds)
        result = defaultdict(list)
        for batch, one_batch in enumerate(ds.get_batch_iter()):
            one_result = self.model.eval_one_step(mode, batch, one_batch)
            for k, v in one_result.items():
                result[k].append(v)
        result = self.model.eval_callback(mode, ds, result)
        self.prev_dev_result = result

        self.logger.write_result(result, time=time.time() - start_time,
                                 is_dev=(mode == 'dev'), is_test=(mode == 'test'))
        return result

    def load(self):
        # write to self.train_ds, self.dev_ds, self.test_ds
        raise NotImplementedError

    def load_train(self, o_load_train):
        raise NotImplementedError
        """recovery from training

        o_load_train can be:
            - <path to status.json>
            - <path to status.json>!-n
        n is number to choose which checkpoint you want use.
        -1 is the last one. so on.
        """
        if '!' in o_load_train:
            *path, index = o_load_train.split('!')
            path = '!'.join(path)
        else:
            path = o_load_train
            index = -1

        with open(path) as f:
            status_dict = json.load(f)

        self.best_epoch = status_dict['best_epoch']
        self.checkpoints = status_dict['checkpoints']

    def load_model(self, o_load_model):
        self.model.load(o_load_model)

    def save_checkpoint(self, epoch):
        save_path = self.checkpoints_path + f'/{epoch}'
        make_sure_dir_exists(save_path)
        self.model.save(save_path)
        self.checkpoints.append(save_path)

        if len(self.checkpoints) > self.o.save_checkpoint_max:
            to_delete = self.checkpoints.pop()
            shutil.rmtree(to_delete)

        status_dict = {'best_epoch': self.best_epoch, 'checkpoints': self.checkpoints}
        try:
            json.dumps(self.best)
        except TypeError:
            pickle_path = save_path + f'/best.pkl'
            pickle.dump(self.best, pickle_path)
            status_dict['best'] = pickle_path
        else:
            status_dict['best'] = self.best

        with open(self.workspace + '/status.json', 'w') as f:
            json.dump(status_dict, f)

    def start(self):
        if self.o.max_epoch:
            try:
                self.train()
            except KeyboardInterrupt:
                if self.o.run_test:
                    print()
                    self.evaluate('test')
        else:
            if self.o.run_dev:
                self.evaluate('dev')
            if self.o.run_test:
                self.evaluate('test')

    def clean(self):
        shutil.rmtree(self.workspace)
