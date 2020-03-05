import os
import dataclasses
from utils.common import *
from utils.functions import calculate_uas, cp2torch, print_to_file, set_seed, torch2cp, use_torch_in_cupy_malloc
from utils.runner import Runner, Model, Logger, RunnerOptions
from utils.data import ConllDataset, WSJ_POS
from module.dmv import DMV, DMVOptions
from module.neural_m import NeuralM, NeuralMOptions


@dataclasses.dataclass
class NMDVModelOptions(RunnerOptions, DMVOptions, NeuralMOptions):
    save_parse_result: bool = False

    dmv_batch_size: int = 10000
    reset_neural: bool = True
    neural_stop_criteria: float = 0.05
    neural_max_subepoch: int = 50
    neural_init_epoch: int = 1

    # overwrite default opotions
    train_ds: str = 'data/wsj10_tr'
    dev_ds: str = 'data/wsj10_d'
    test_ds: str = 'data/wsj10_te'
    num_tag: int = 34
    max_len: int = 10

    dim_pos_emb: int = 20
    dim_valence_emb: int = 10
    dim_hidden: int = 20
    dim_pre_out_decision: int = 7
    dim_pre_out_child: int = 10
    dropout: float = 0.1
    lr: float = 0.01
    use_pos_emb: bool = True
    use_valence_emb: bool = True

    batch_size: int = 256
    max_epoch: int = 50
    early_stop: int = 10
    compare_field: str = 'likelihood'
    save_best: bool = True
    show_log: bool = True
    save_log: bool = True

    run_dev: bool = True
    run_test: bool = True

    e_step_mode: str = 'viterbi'
    cv: int = 2
    count_smoothing: float = 0.1
    param_smoothing: float = 0.1


class NDMVModel(Model):
    def __init__(self, o: NMDVModelOptions, r: Runner):
        self.o = o
        self.r = r

        # store train_data param when eval
        self.model_dec_params = None
        self.model_trans_params = None

    def build(self):
        self.dmv = DMV(self.o)
        self.dmv.init_specific(self.r.train_ds.get_len())
        self.neural_m = NeuralM(self.o).cuda()

    def train_init(self, epoch_id, dataset):
        dataset.build_batchs(self.o.dmv_batch_size, False, True)
        if self.o.reset_neural:
            self.neural_m.reset()
            # self.neural_m = NeuralM(self.o).cuda()
        if self.dmv.initializing and epoch_id >= self.o.neural_init_epoch:
            self.dmv.initializing = False
            self.r.logger.write("finishing initialization")
        self.dmv.reset_root_counter()

    def train_one_step(self, epoch_id, batch_id, one_batch):
        batch_size = len(one_batch[0])

        idx = np.arange(batch_size)

        id_array = one_batch[0]
        pos_array = cpasarray(one_batch[1])
        len_array = one_batch[3]

        ll = self.dmv.e_step(id_array, pos_array, len_array)
        self.dmv.batch_dec_trace = cp.sum(self.dmv.batch_dec_trace, axis=2)

        dec_trace = cp2torch(self.dmv.batch_dec_trace)
        trans_trace = cp2torch(self.dmv.batch_trans_trace)
        pos_array = cp2torch(pos_array)
        len_array_gpu = torch.tensor(len_array, device='cuda')

        self.neural_m.train()
        loss_previous = 0.
        for sub_run in range(self.o.neural_max_subepoch):
            loss_current = 0.

            np.random.shuffle(idx)
            for i in range(0, batch_size, self.o.batch_size):
                self.neural_m.optimizer.zero_grad()
                sub_idx = slice(i, i + self.o.batch_size)
                # sub_idx = idx[i: i + self.o.batch_size]
                arrays = {'pos': pos_array[sub_idx], 'len': len_array_gpu[sub_idx]}
                traces = {'decision': dec_trace[sub_idx], 'transition': trans_trace[sub_idx]}

                loss = self.neural_m(arrays, pos_array[sub_idx], traces=traces)
                loss_current += loss.item()
                loss.backward()
                self.neural_m.optimizer.step()

            if loss_previous > 0.:
                diff_rate = abs(loss_previous - loss_current) / loss_previous
                if diff_rate < self.o.neural_stop_criteria:
                    break
            loss_previous = loss_current

        self.neural_m.eval()
        with torch.no_grad():
            for i in range(0, batch_size, self.o.batch_size):
                sub_idx = slice(i, i + self.o.batch_size)
                sub_id_array = id_array[sub_idx]
                sub_len_array = len_array[sub_idx]
                arrays = {'pos': pos_array[sub_idx], 'len': len_array_gpu[sub_idx]}

                dec_param, trans_param = self.neural_m(arrays, pos_array[sub_idx])
                dec_param = dec_param.cpu().numpy()
                trans_param = trans_param.cpu().numpy()
                self.dmv.put_decision_param(sub_id_array, dec_param, sub_len_array)
                self.dmv.put_transition_param(sub_id_array, trans_param, sub_len_array)

        return {'loss': loss_current, 'likelihood': ll, 'runs': sub_run + 1}

    def train_callback(self, epoch_id, dataset, result):
        self.dmv.m_step()
        return {'loss': sum(result['loss']) / len(result['loss']),
                'likelihood': sum(result['likelihood']),
                'runs': sum(result['runs']) / len(result['runs'])}

    def eval_init(self, mode, dataset):
        if mode == 'test' and self.r.best is not None and self.o.save_best:
            self.load(self.r.best_path)

        # backup train status
        self.model_dec_params = self.dmv.all_dec_param
        self.model_trans_params = self.dmv.all_trans_param

        # init eval status
        self.neural_m.eval()
        self.dmv.init_specific(dataset.get_len())

    def eval_one_step(self, mode, batch_id, one_batch):
        with torch.no_grad():
            pos_array = torch.tensor(one_batch[1], dtype=torch.long, device='cuda')
            len_array = torch.tensor(one_batch[3], dtype=torch.long, device='cuda')

            arrays = {'pos': pos_array, 'len': len_array}
            dec_param, trans_param = self.neural_m(arrays, pos_array)

            id_array = one_batch[0]
            len_array = one_batch[3]
            dec_param = dec_param.cpu().numpy()
            trans_param = trans_param.cpu().numpy()
            self.dmv.put_decision_param(id_array, dec_param, len_array)
            self.dmv.put_transition_param(id_array, trans_param, len_array)

            pos_array = torch2cp(pos_array)
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
            print_to_file(result, self.dev_data, os.path.join(self.r.workspace, 'parsed.txt'))

        # restore train status
        self.dmv.all_dec_param = self.model_dec_params
        self.dmv.all_trans_param = self.model_trans_params
        return {'uas': acc * 100, 'likelihood': ll}

    def init_param(self, dataset):
        dataset.build_batchs(self.o.batch_size, same_len=True)
        self.dmv.init_param(dataset)

    def save(self, folder_path):
        self.dmv.save(folder_path)
        self.neural_m.save(folder_path)

    def load(self, folder_path):
        self.dmv.load(folder_path)
        self.neural_m.load(folder_path)

    def __str__(self):
        return f'NDMV_{self.o.e_step_mode}_{self.o.cv}'


class NDMVModelRunner(Runner):
    def __init__(self, o):
        if o.use_softmax_em:
            from utils import common
            common.cpf = cp.float64

        m = NDMVModel(o, self)
        super().__init__(m, o, Logger(o))

    def load(self):
        self.train_ds = ConllDataset(self.o.train_ds, pos_vocab=WSJ_POS, sort=True)
        self.dev_ds = ConllDataset(self.o.dev_ds, pos_vocab=WSJ_POS)
        self.test_ds = ConllDataset(self.o.test_ds, pos_vocab=WSJ_POS)

        self.dev_ds.build_batchs(self.o.batch_size)
        self.test_ds.build_batchs(self.o.batch_size)


if __name__ == '__main__':
    use_torch_in_cupy_malloc()
    options = NMDVModelOptions()
    options.parse()
    runner = NDMVModelRunner(options)
    runner.start()
