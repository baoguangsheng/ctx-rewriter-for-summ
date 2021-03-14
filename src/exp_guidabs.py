import argparse
import os
import os.path as path
import glob
import torch
import torch.nn as nn
import logging
from others.logging import set_logger, init_logger
import train_abstractive as train_abs
import models.data_loader as data_ldr
import models.model_builder as model_bld
import models.predictor as pred_abs
from pytorch_transformers import BertTokenizer
from exp_base import test_ext_abs, test_abs, test_ext

logger = logging.getLogger(__name__)

''' Main experiment to build GuidAbs (ContextRewriter)
'''


def _top_model(model_path, n=0):
    top = list(glob.glob(path.join(model_path, 'top%s.model_step_*.pt' % n)))
    assert len(top) == 1, 'Unexpected matches for top model: %s' % top
    return top[0]


class GuidAbsHandler:
    def __init__(self, root_path, quick_test=False, max_pos=512, word_dropout=0.0, sent_dropout=0.0):
        self.data_path = './bert_data'
        self.model_path = path.join(root_path, 'models_guidabs')
        self.quick_test = quick_test
        self.max_pos = max_pos
        self.word_dropout = word_dropout
        self.sent_dropout = sent_dropout

    def _build_abs_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-task", default='abs')
        parser.add_argument("-encoder", default='bert')
        parser.add_argument("-mode", default='train')
        parser.add_argument("-bert_model_path", default='./bert_pretrained/')
        parser.add_argument("-bert_data_path", default='./bert_data/cnndm')
        parser.add_argument("-model_path", default='./models/')
        parser.add_argument("-result_path", default='./results/cnndm')
        parser.add_argument('-log_file', default='./logs/exp_guidabs.log')
        parser.add_argument("-temp_dir", default='./temp')

        if self.quick_test:
            parser.add_argument("-train_steps", default=200, type=int)
            parser.add_argument("-warmup_steps_bert", default=100, type=int)
            parser.add_argument("-warmup_steps_dec", default=100, type=int)
            parser.add_argument("-report_every", default=5, type=int)
            parser.add_argument("-save_checkpoint_steps", default=50, type=int)
            parser.add_argument("-test_start_from", default=20, type=int)
            parser.add_argument("-batch_size", default=100, type=int)
            parser.add_argument("-test_batch_size", default=2000, type=int)
            parser.add_argument("-accum_count", default=1, type=int)
            parser.add_argument('-visible_gpus', default='0', type=str)
        else:
            parser.add_argument("-train_steps", default=240000, type=int)
            parser.add_argument("-warmup_steps_bert", default=20000, type=int)
            parser.add_argument("-warmup_steps_dec", default=10000, type=int)
            parser.add_argument("-report_every", default=50, type=int)
            parser.add_argument("-save_checkpoint_steps", default=2000, type=int)
            parser.add_argument("-test_start_from", default=120000, type=int)
            parser.add_argument("-batch_size", default=560, type=int)
            parser.add_argument("-test_batch_size", default=3000, type=int)
            parser.add_argument("-accum_count", default=3, type=int)
            parser.add_argument('-visible_gpus', default='0,1', type=str)

        parser.add_argument("-max_pos", default=self.max_pos, type=int)
        parser.add_argument("-lr_bert", default=0.002, type=float)
        parser.add_argument("-lr_dec", default=0.2, type=float)
        parser.add_argument("-sep_optim", default=True, type=bool)
        parser.add_argument("-use_bert_emb", default=True, type=bool)
        parser.add_argument("-use_interval", default=True, type=bool)
        parser.add_argument("-large", default=False, type=bool)

        parser.add_argument("-finetune_bert", default=True, type=bool)
        parser.add_argument("-dec_dropout", default=0.2, type=float)
        parser.add_argument("-dec_layers", default=6, type=int)
        parser.add_argument("-dec_hidden_size", default=768, type=int)
        parser.add_argument("-dec_heads", default=8, type=int)
        parser.add_argument("-dec_ff_size", default=2048, type=int)

        parser.add_argument("-label_smoothing", default=0.1, type=float)
        parser.add_argument("-generator_shard_size", default=32, type=int)
        parser.add_argument("-alpha", default=0.95, type=float)
        parser.add_argument("-beam_size", default=5, type=int)
        parser.add_argument("-min_length", default=50, type=int)
        parser.add_argument("-max_length", default=200, type=int)
        parser.add_argument("-max_tgt_len", default=140, type=int)

        parser.add_argument("-param_init", default=0, type=float)
        parser.add_argument("-param_init_glorot", default=True, type=bool)
        parser.add_argument("-optim", default='adam', type=str)
        parser.add_argument("-beta1", default=0.9, type=float)
        parser.add_argument("-beta2", default=0.999, type=float)
        parser.add_argument("-max_grad_norm", default=0, type=float)

        parser.add_argument('-gpu_ranks', default='0', type=str)
        parser.add_argument('-seed', default=666, type=int)

        parser.add_argument("-test_all", default=True, type=bool)
        parser.add_argument("-test_from", default='')
        parser.add_argument("-train_from", default='')
        parser.add_argument("-recall_eval", default=False, type=bool)
        parser.add_argument("-report_rouge", default=True, type=bool)
        parser.add_argument("-block_trigram", default=True, type=bool)

        # guide tags
        parser.add_argument("-max_n_tags", default=6, type=int)
        parser.add_argument("-tag_dropout", default=0.2, type=float)
        parser.add_argument("-word_dropout", default=self.word_dropout, type=float)
        parser.add_argument("-sent_dropout", default=self.sent_dropout, type=float)

        args = parser.parse_args('')
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
        args.world_size = len(args.gpu_ranks)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

        args.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        args.device_id = 0 if args.device == "cuda" else -1
        return args

    def train_model(self):
        logger.info('Train GuidAbs model %s' % self.model_path)
        fn_touch = path.join(self.model_path, 'finished.train_guidabs_model')
        if path.exists(fn_touch):
            return
        if not path.exists(self.model_path):
            os.mkdir(self.model_path)
        args = self._build_abs_args()
        args.mode = 'train'
        args.bert_data_path = path.join(self.data_path, 'cnndm')
        args.model_path = self.model_path
        args.log_file = path.join(self.model_path, 'abs_bert_cnndm.log')
        init_logger(args.log_file)
        train_abs.train_abs(args, args.device_id)
        os.system('touch %s' % fn_touch)

    def validate_model(self):
        logger.info('Validate GuidAbs model %s' % self.model_path)
        fn_touch = path.join(self.model_path, 'finished.validate_guidabs_model')
        if path.exists(fn_touch):
            return
        args = self._build_abs_args()
        args.mode = 'validate'
        args.bert_data_path = path.join(self.data_path, 'cnndm')
        args.model_path = self.model_path
        args.log_file = path.join(self.model_path, 'val_abs_bert_cnndm.log')
        args.batch_size = args.test_batch_size
        init_logger(args.log_file)
        acc_top3 = train_abs.validate_abs(args, args.device_id)
        # rename top3 models and remove other models
        for i, (acc, xent, cp) in enumerate(acc_top3):
            fn = path.basename(cp)
            tgt_path = path.join(self.model_path, 'top%s.%s' % (i, fn))
            os.system('mv %s %s -f' % (cp, tgt_path))
            logger.info('Archive validated GuidAbs model %s' % tgt_path)
        os.system('rm %s/model_step_*.pt -f' % self.model_path)
        os.system('touch %s' % fn_touch)

    def test_model(self, corpus_type, topn=0):
        model_file = _top_model(self.model_path, n=topn)
        logger.info('Test GuidAbs model %s' % model_file)
        fn_touch = path.join(self.model_path, 'finished_%s.test_guidabs_model%s' % (corpus_type, topn))
        if path.exists(fn_touch):
            return
        args = self._build_abs_args()
        args.mode = 'test'
        args.bert_data_path = path.join(self.data_path, 'cnndm')
        args.model_path = self.model_path
        args.log_file = path.join(self.model_path, 'test_abs_bert_cnndm_%s_top%s.log' % (corpus_type, topn))
        args.result_path = path.join(self.model_path, 'cnndm_%s_top%s' % (corpus_type, topn))
        init_logger(args.log_file)
        step = int(model_file.split('.')[-2].split('_')[-1])
        # load abs model
        step_abs = int(model_file.split('.')[-2].split('_')[-1])
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model_abs = model_bld.AbsSummarizer(args, args.device, checkpoint)
        model_abs.eval()
        # init model testers
        tokenizer = BertTokenizer.from_pretrained(path.join(args.bert_model_path, model_abs.bert.model_name),
                                                  do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

        predictor = pred_abs.build_predictor(args, tokenizer, symbols, model_abs, logger)
        test_iter = data_ldr.Dataloader(args, data_ldr.load_dataset(args, corpus_type, shuffle=False),
                                            args.test_batch_size, args.device,
                                            shuffle=False, is_test=True, keep_order=True)

        avg_f1 = test_abs(logger, args, predictor, step_abs, test_iter)
        os.system('touch %s' % fn_touch)
        return avg_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # split bert data
    parser.add_argument("-source_data_path", default='./bert_data/')
    parser.add_argument("-root_path", default='./exp_guidabs')
    parser.add_argument("-log_file", default='./logs/exp_guidabs.log')
    parser.add_argument("-quick_test", nargs='?', const=True, default=False, type=bool)
    args = parser.parse_args()

    set_logger(logger, args.log_file)

    if not os.path.exists(args.root_path):
        os.mkdir(args.root_path)
    abs = GuidAbsHandler(args.root_path, args.quick_test, max_pos=512, word_dropout=0.3, sent_dropout=0.2)
    abs.train_model()
    abs.validate_model()
    # choose the model with the best average ROUGE score
    abs_best_i = None
    abs_best_f1 = None
    for i in range(3):
        avg_f1 = abs.test_model('valid', i)
        if abs_best_f1 is None or avg_f1 > abs_best_f1:
            abs_best_f1 = avg_f1
            abs_best_i = i
    logger.info('The best GuidAbs model is top%s with an average rouge f1 of %s.' % (abs_best_i, abs_best_f1))
    # test the model with Oracle extractive
    abs.test_model('test', abs_best_i)
