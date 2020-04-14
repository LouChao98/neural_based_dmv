import os
import dataclasses
from utils.common import *
from utils.functions import *
from utils.runner import Runner, Model, Logger, RunnerOptions
from utils.data import ConllDataset, Vocab, WSJ_POS
from module.dmv import DMV, DMVOptions
from module.neural_m import NeuralM, NeuralMOptions


@dataclasses.dataclass
class DNMDVModelOptions(RunnerOptions, DMVOptions, NeuralMOptions):
    save_parse_result: bool = False
    vocab_path: str = 'data/bllip_vec/pos.txt'
    # pos_path: str = 'data/bllip_vec/pos.txt'

    emb_path: str = ''#data/bllip_vec/posvec.npy'
    pos_emb_path: str = ''
    # emb_path: str = 'data/bllip_vec/sub_wordvectors.npy'
    # pos_emb_path: str = 'data/bllip_vec/posvec.npy'

    dmv_batch_size: int = 10240
    reset_neural: bool = False
    neural_stop_criteria: float = 0.001
    neural_max_subepoch: int = 100
    neural_init_epoch: int = 1

    # overwrite default opotions
    train_ds: str = 'data/wsj10_tr'
    dev_ds: str = 'data/wsj10_d'
    test_ds: str = 'data/wsj10_te'
    pretrained_ds: str = 'data/wsj10_tr_pred'
    num_tag: int = 35
    max_len: int = 10

    # dim_pos_emb: int = 20
    dim_word_emb: int = 20
    dim_valence_emb: int = 10  # 30??
    dim_hidden: int = 32
    dim_pre_out_decision: int = 12
    dim_pre_out_child: int = 24
    dropout: float = 0.2
    lr: float = 0.001
    use_pos_emb: bool = False
    use_word_emb: bool = True
    use_sentence_emb: bool = True
    use_valence_emb: bool = True
    encoder_mode: str = 'lstm'
    encoder_lstm_dim_hidden: int = 16
    encoder_lstm_num_layers: int = 1
    encoder_lstm_dropout: float = 0.

    batch_size: int = 512
    max_epoch: int = 100
    early_stop: int = 20
    compare_field: str = 'likelihood'
    save_best: bool = True
    show_log: bool = True
    save_log: bool = True
    show_best_hit: bool = True

    run_dev: bool = True
    run_test: bool = True

    e_step_mode: str = 'viterbi'
    cv: int = 2
    count_smoothing: float = 0.1
    param_smoothing: float = 0.1


class DNDMVModel(Model):
    def __init__(self, o: DNMDVModelOptions, r: 'DNDMVModelRunner'):
        self.o = o
        self.r = r

        # store train_data param when eval
        self.model_dec_params = None
        self.model_trans_params = None

    def build(self, nn_only=False):
        if not nn_only:
            self.dmv = DMV(self.o)
            self.dmv.init_specific(self.r.train_ds.get_len())
        self.neural_m = NeuralM(self.o, self.r.word_emb, None, None).cuda()

    def train_init(self, epoch_id, dataset):
        dataset.build_batchs(self.o.dmv_batch_size, False, True)
        if self.o.reset_neural:
            self.build(nn_only=True)
        if self.dmv.initializing:
            self.r.best = None
            self.r.best_epoch = -1
        if self.dmv.initializing and epoch_id >= self.o.neural_init_epoch:
            self.dmv.initializing = False
            self.r.logger.write("finishing initialization")
        self.dmv.reset_root_counter()

    def train_one_step(self, epoch_id, batch_id, one_batch):
        # TODO use nn predict ROOT param
        batch_size = len(one_batch[0])  # self.o.dmv_batch_size
        idx = np.arange(batch_size)

        id_array = one_batch[0]
        pos_array = cpasarray(one_batch[1])
        pos_array_torch = cp2torch(pos_array)
        len_array = one_batch[3]
        len_array_gpu = torch.tensor(len_array, dtype=torch.long, device='cuda')
        max_len = np.max(len_array)

        if self.dmv.initializing:
            ll = self.dmv.e_step(id_array, pos_array, len_array)
        else:
            # TODO clean code
            self.neural_m.eval()
            trans_params, dec_params = [], []
            with torch.no_grad():
                for i in range(0, batch_size, self.o.batch_size):
                    sub_idx = slice(i, i + self.o.batch_size)
                    arrays = {'word': pos_array_torch[sub_idx], 'len': len_array_gpu[sub_idx]}

                    dec_param, trans_param = self.neural_m(arrays, pos_array_torch[sub_idx])
                    trans_params.append(trans_param)
                    dec_params.append(dec_param)

            trans_param = cpfempty((batch_size, max_len + 1, max_len + 1, self.o.cv))
            dec_param = cpfempty((batch_size, max_len + 1, 2, 2, 2))
            offset = 0
            for t, d in zip(trans_params, dec_params):
                _, batch_len, *_ = t.shape
                t = torch2cp(t)
                d = torch2cp(d)
                trans_param[offset: offset + self.o.batch_size, 1:batch_len + 1, 1:batch_len + 1] = t
                dec_param[offset: offset + self.o.batch_size, 1:batch_len + 1] = d
                offset += self.o.batch_size
            root_param = cp.expand_dims(self.dmv.root_param, 0)
            root_scores = cp.expand_dims(cp.take_along_axis(root_param, self.dmv.input_gaurd(pos_array), 1), -1)
            trans_param[:, 0, :, :] = root_scores
            trans_param[:, :, 0, :] = -cp.inf

            ll = self.dmv.e_step_using_unmnanaged_score(pos_array, len_array, trans_param, dec_param)
        self.dmv.batch_dec_trace = cp.sum(self.dmv.batch_dec_trace, axis=2)

        dec_trace = cp2torch(self.dmv.batch_dec_trace)
        trans_trace = cp2torch(self.dmv.batch_trans_trace)

        self.neural_m.train()
        loss_previous = 0.
        for sub_run in range(self.o.neural_max_subepoch):
            loss_current = 0.

            np.random.shuffle(idx)
            for i in range(0, batch_size, self.o.batch_size):
                self.neural_m.optimizer.zero_grad()
                # sub_idx = slice(i, i + self.o.batch_size)
                sub_idx = idx[i: i + self.o.batch_size]
                arrays = {'word': pos_array_torch[sub_idx], 'len': len_array_gpu[sub_idx]}
                traces = {'decision': dec_trace[sub_idx], 'transition': trans_trace[sub_idx]}

                loss = self.neural_m(arrays, pos_array_torch[sub_idx], traces=traces)
                loss_current += loss.item()
                loss.backward()
                self.neural_m.optimizer.step()

            if loss_previous > 0.:
                diff_rate = abs(loss_previous - loss_current) / loss_previous
                if diff_rate < self.o.neural_stop_criteria:
                    break
            loss_previous = loss_current

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

            arrays = {'word': pos_array, 'len': len_array}
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
            print_to_file(result, self.r.dev_ds, os.path.join(self.r.workspace, 'parsed.txt'))

        # restore train status
        self.dmv.all_dec_param = self.model_dec_params
        self.dmv.all_trans_param = self.model_trans_params
        return {'uas': acc * 100, 'likelihood': ll}

    def init_param(self, dataset):
        dataset.build_batchs(self.o.batch_size, same_len=True)
        if self.r.pretrained_ds:
            self.dmv.init_pretrained(self.r.pretrained_ds)
        else:
            dataset.build_batchs(self.o.batch_size, same_len=True)
            self.dmv.init_param(dataset)

    def save(self, folder_path):
        self.dmv.save(folder_path)
        self.neural_m.save(folder_path)

    def load(self, folder_path):
        self.dmv.load(folder_path)
        self.neural_m.load(folder_path)

    def __str__(self):
        return f'DNDMV_{self.o.e_step_mode}_{self.o.cv}'


class DNDMVModelRunner(Runner):
    def __init__(self, o: DNMDVModelOptions):
        if o.use_softmax_em:
            from utils import common
            common.cpf = cp.float64

        m = DNDMVModel(o, self)
        super().__init__(m, o, Logger(o))

    def load(self):
        pos_vocab_list = [w.strip() for w in open(self.o.vocab_path)]
        pos_vocab = Vocab.from_list(pos_vocab_list)
        self.train_ds = ConllDataset(self.o.train_ds, pos_vocab=pos_vocab)
        self.dev_ds = ConllDataset(self.o.dev_ds, pos_vocab=pos_vocab)
        self.test_ds = ConllDataset(self.o.test_ds, pos_vocab=pos_vocab)

        if self.o.pretrained_ds:
            self.pretrained_ds = ConllDataset(self.o.pretrained_ds, pos_vocab=pos_vocab)
        else:
            self.pretrained_ds = None

        self.dev_ds.build_batchs(self.o.batch_size)
        self.test_ds.build_batchs(self.o.batch_size)

        if self.o.emb_path:
            self.word_emb = np.load(self.o.emb_path)
        else:
            self.word_emb = None


if __name__ == '__main__':
    use_torch_in_cupy_malloc()
    options = DNMDVModelOptions()
    options.parse()
    runner = DNDMVModelRunner(options)
    if options.pretrained_ds:
        runner.logger.write('init with acc:')
        runner.evaluate('dev')
        runner.evaluate('test')
    runner.start()
