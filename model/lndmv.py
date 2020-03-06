import os
import dataclasses
from utils.common import *
from utils.functions import calculate_uas, cp2torch, get_init_param_converter, get_tag_id_converter, print_to_file, \
    set_seed, torch2cp, use_torch_in_cupy_malloc
from utils.runner import Runner, Model, Logger, RunnerOptions
from utils.data import ConllDataset, Vocab, BLLIP_POS
from module.dmv import DMV, DMVOptions
from module.neural_m import NeuralM, NeuralMOptions


@dataclasses.dataclass
class LNMDVModelOptions(RunnerOptions, DMVOptions, NeuralMOptions):
    save_parse_result: bool = False
    vocab_path: str = 'data/bllip_vec/vocab.txt'
    pos_path: str = 'data/bllip_vec/pos.txt'
    emb_path: str = 'data/bllip_vec/sub_wordvectors.npy'

    # dim=dim_word_emb. to build `pre_out_child` matrix when use_emb_as_w=True
    # NOT for converting pos array to vectors
    pos_emb_path: str = 'data/bllip_vec/posvectors.npy'

    dmv_batch_size: int = 10240
    reset_neural: bool = False
    neural_stop_criteria: float = 0.001
    neural_max_subepoch: int = 50
    neural_init_epoch: int = 1

    pretrained_ds = 'data/wsj10_tr_pred'

    # overwrite default opotions
    train_ds: str = 'data/bllip_conll/bllip10clean_20k.conll'
    dev_ds: str = 'data/wsj10_d_retag'
    test_ds: str = 'data/wsj10_te_retag'
    num_lex: int = 390  # not include <UNK> <PAD>

    dim_pos_emb: int = 20
    dim_word_emb: int = 100
    dim_valence_emb: int = 20
    dim_hidden: int = 128
    dim_pre_out_decision: int = 32
    dim_pre_out_child: int = 100
    dropout: float = 0.3
    lr: float = 0.01
    use_pos_emb: bool = True
    use_word_emb: bool = True
    use_valence_emb: bool = True
    use_emb_as_w: bool = True
    freeze_word_emb: bool = True

    batch_size: int = 2048
    max_epoch: int = 100
    early_stop: int = 10
    compare_field: str = 'likelihood'
    save_best: bool = True
    show_log: bool = True
    show_best_hit: bool = True

    run_dev: bool = True
    run_test: bool = True

    e_step_mode: str = 'viterbi'
    cv: int = 2
    count_smoothing: float = 0.1
    param_smoothing: float = 0.1


class LNDMVModel(Model):
    def __init__(self, o: LNMDVModelOptions, r: Runner):
        self.o = o
        self.r = r

        # store train_data param when eval
        self.model_dec_params = None
        self.model_trans_params = None

    def build(self, nn_only=False):
        word_idx = cp.arange(2, len(self.r.train_ds.word_vocab))[:self.o.num_lex]
        if not nn_only:
            self.dmv = DMV(self.o)
            self.dmv.init_specific(self.r.train_ds.get_len())
            self.converter = get_tag_id_converter(word_idx, len(self.r.train_ds.pos_vocab))

        self.neural_m = NeuralM(self.o, self.r.word_emb, self.r.pos_emb).cuda()
        self.neural_m.set_lex(cp2torch(word_idx), None)

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
        batch_size = len(one_batch[0])

        idx = np.arange(batch_size)

        id_array = one_batch[0]
        pos_array = cpasarray(one_batch[1])
        word_array = cpasarray(one_batch[2])
        len_array = one_batch[3]
        tag_array = self.converter(word_array, pos_array)

        ll = self.dmv.e_step(id_array, tag_array, len_array)
        self.dmv.batch_dec_trace = cp.sum(self.dmv.batch_dec_trace, axis=2)

        dec_trace = cp2torch(self.dmv.batch_dec_trace)
        trans_trace = cp2torch(self.dmv.batch_trans_trace)
        pos_array = cp2torch(pos_array)
        word_array = cp2torch(word_array)
        tag_array = cp2torch(tag_array)
        len_array_gpu = torch.tensor(len_array, device='cuda')

        self.neural_m.train()
        loss_previous = 0.
        for sub_run in range(self.o.neural_max_subepoch):
            loss_current = 0.
            np.random.shuffle(idx)
            for i in range(0, batch_size, self.o.batch_size):
                self.neural_m.optimizer.zero_grad()
                # sub_idx = slice(i, i + self.o.batch_size)
                sub_idx = idx[i: i + self.o.batch_size]
                arrays = {'pos': pos_array[sub_idx], 'word': word_array[sub_idx], 'len': len_array_gpu[sub_idx]}
                traces = {'decision': dec_trace[sub_idx], 'transition': trans_trace[sub_idx]}

                loss = self.neural_m(arrays, tag_array[sub_idx], traces=traces)
                loss_current += loss.item()
                loss.backward()
                self.neural_m.optimizer.step()

            if loss_previous > 0. and not self.dmv.initializing:
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
                arrays = {'pos': pos_array[sub_idx], 'word': word_array[sub_idx], 'len': len_array_gpu[sub_idx]}

                dec_param, trans_param = self.neural_m(arrays, tag_array[sub_idx])
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
            pos_array = cpasarray(one_batch[1])
            word_array = cpasarray(one_batch[2])
            tag_array = self.converter(word_array, pos_array)

            pos_array = cp2torch(pos_array)
            word_array = cp2torch(word_array)

            len_array = torch.tensor(one_batch[3], device='cuda')

            arrays = {'pos': pos_array, 'word': word_array, 'len': len_array}
            dec_param, trans_param = self.neural_m(arrays, cp2torch(tag_array))

            id_array = one_batch[0]
            len_array = one_batch[3]
            dec_param = dec_param.cpu().numpy()
            trans_param = trans_param.cpu().numpy()
            self.dmv.put_decision_param(id_array, dec_param, len_array)
            self.dmv.put_transition_param(id_array, trans_param, len_array)

            out = self.dmv.parse(id_array, tag_array, len_array)
            out['likelihood'] = self.dmv.e_step(id_array, tag_array, len_array)
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
        word_idx = cp.arange(2, len(dataset.word_vocab))
        converter = get_init_param_converter(word_idx, len(dataset.pos_vocab))
        if self.r.pretrained_ds:
            self.dmv.init_pretrained(self.r.pretrained_ds, converter)
        else:
            dataset.build_batchs(self.o.batch_size, same_len=True)
            self.dmv.init_param(dataset, converter)

    def save(self, folder_path):
        self.dmv.save(folder_path)
        self.neural_m.save(folder_path)

    def load(self, folder_path):
        self.dmv.load(folder_path)
        self.neural_m.load(folder_path)

    def __str__(self):
        return f'LNDMV_{self.o.e_step_mode}_{self.o.cv}_{len(self.r.train_ds.word_vocab)-2}'


class LNDMVModelRunner(Runner):
    def __init__(self, o: LNMDVModelOptions):
        if o.use_softmax_em:
            from utils import common
            common.cpf = cp.float64

        m = LNDMVModel(o, self)
        super().__init__(m, o, Logger(o))

    def load(self):
        word_vocab_list = [w.strip() for w in open(self.o.vocab_path)][:self.o.num_lex + 2]
        word_vocab = Vocab.from_list(word_vocab_list, unk='<UNK>', pad='<PAD>')

        self.train_ds = ConllDataset(self.o.train_ds, pos_vocab=BLLIP_POS, word_vocab=word_vocab)
        self.dev_ds = ConllDataset(self.o.dev_ds, pos_vocab=BLLIP_POS, word_vocab=word_vocab)
        self.test_ds = ConllDataset(self.o.test_ds, pos_vocab=BLLIP_POS, word_vocab=word_vocab)

        if self.o.pretrained_ds:
            self.pretrained_ds = ConllDataset(self.o.pretrained_ds, pos_vocab=BLLIP_POS, word_vocab=word_vocab)
        else:
            self.pretrained_ds = None

        self.dev_ds.build_batchs(self.o.batch_size)
        self.test_ds.build_batchs(self.o.batch_size)

        self.o.max_len = 10
        self.o.num_tag = len(BLLIP_POS) + self.o.num_lex

        if self.o.emb_path:
            self.word_emb = np.load(self.o.emb_path)[:self.o.num_lex + 2]
        else:
            self.word_emb = None
        self.pos_emb = np.load(self.o.pos_emb_path) if self.o.pos_emb_path else None


if __name__ == '__main__':
    use_torch_in_cupy_malloc()

    options = LNMDVModelOptions()
    options.parse()

    runner = LNDMVModelRunner(options)
    if options.pretrained_ds:
        runner.logger.write('init with acc:')
        runner.evaluate('test')
    runner.start()
