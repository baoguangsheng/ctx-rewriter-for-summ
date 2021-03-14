import argparse
import os
import os.path as path
import numpy as np
import glob
import torch
import logging
from others.logging import set_logger, init_logger
import models.data_loader as data_ldr
import models.model_builder as model_bld
import models.predictor as pred_abs
from pytorch_transformers import BertTokenizer
from exp_base import test_ext_abs, test_abs, test_ext, ExtDecider

logger = logging.getLogger(__name__)

''' Experiment GuidAbs with various extractive model
'''

class Lead3Extractor:
    def __init__(self):
        self.name = 'LEAD3'
        self.max_n_tags = 6

    def extract(self, batch):
        src_str = batch.src_str
        res_src = ['<q>'.join(doc[:3]) for doc in src_str]

        segs = batch.segs
        src_tags = np.zeros((segs.size(0), segs.size(1), self.max_n_tags - 1), dtype=np.int)
        segs = segs.cpu().data.numpy()
        src_tags[:, :, 0] = (segs == 1).astype(np.float)
        src_tags[:, :, 1] = (segs == 2).astype(np.float)
        src_tags[:, :, 2] = (segs == 3).astype(np.float)
        res_tags = torch.tensor(src_tags, dtype=torch.float).to(batch.src.device)
        return res_src, res_tags


class BertSumExtractor:
    def __init__(self, ext_model_file, block_trigram=True):
        import presumm.model_builder as presumm_model
        import presumm.trainer_ext as presumm_trainer_ext
        args = self._build_ext_args()
        args.block_trigram = block_trigram
        checkpoint = torch.load(ext_model_file, map_location=lambda storage, loc: storage)
        self.name = 'BERTSUMEXT_blocktrigram' if block_trigram else 'BERTSUMEXT_noblocktrigram'
        self.model_file = ext_model_file
        self.model_ext = presumm_model.ExtSummarizer(args, args.device, checkpoint)
        self.model_ext.eval()
        self.trainer = presumm_trainer_ext.build_trainer(args, args.device_id, self.model_ext, None)

    def _build_ext_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-task", default='ext')
        parser.add_argument("-encoder", default='bert')
        parser.add_argument("-mode", default='train')
        parser.add_argument("-bert_model_path", default='./bert_pretrained/')
        parser.add_argument("-bert_data_path", default='./bert_data/cnndm')
        parser.add_argument("-model_path", default='./models/')
        parser.add_argument("-result_path", default='./results/cnndm')
        parser.add_argument('-log_file', default='./logs/cnndm.log')
        parser.add_argument("-temp_dir", default='./temp')
        parser.add_argument("-train_from", default='')

        parser.add_argument("-max_pos", default=512, type=int)
        parser.add_argument("-max_tgt_len", default=140, type=int)
        parser.add_argument("-max_n_tags", default=6, type=int)
        parser.add_argument("-use_interval", default=True, type=bool)
        parser.add_argument("-large", default=False, type=bool)

        parser.add_argument("-ext_dropout", default=0.2, type=float)
        parser.add_argument("-ext_layers", default=2, type=int)
        parser.add_argument("-ext_hidden_size", default=768, type=int)
        parser.add_argument("-ext_heads", default=8, type=int)
        parser.add_argument("-ext_ff_size", default=2048, type=int)

        parser.add_argument("-param_init", default=0, type=float)
        parser.add_argument("-param_init_glorot", default=True, type=bool)
        parser.add_argument("-optim", default='adam')
        parser.add_argument("-lr", default=2e-3, type=float)
        parser.add_argument("-beta1", default=0.9, type=float)
        parser.add_argument("-beta2", default=0.999, type=float)
        parser.add_argument("-max_grad_norm", default=0, type=float)

        parser.add_argument("-train_steps", default=40000, type=int)
        parser.add_argument("-warmup_steps", default=10000, type=int)
        parser.add_argument("-report_every", default=50, type=int)
        parser.add_argument("-test_start_from", default=10000, type=int)
        parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
        parser.add_argument("-batch_size", default=8*512, type=int)
        parser.add_argument("-accum_count", default=2, type=int)
        parser.add_argument('-visible_gpus', default='0,1', type=str)
        parser.add_argument("-test_batch_size", default=8*512, type=int)

        parser.add_argument("-finetune_bert", default=True, type=bool)
        parser.add_argument('-gpu_ranks', default='0', type=str)
        parser.add_argument('-seed', default=666, type=int)

        parser.add_argument("-test_all", default=True, type=bool)
        parser.add_argument("-test_from", default='')
        parser.add_argument("-recall_eval", default=False, type=bool)
        parser.add_argument("-report_rouge", default=True, type=bool)
        parser.add_argument("-block_trigram", default=True, type=bool)

        args = parser.parse_args('')
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
        args.world_size = len(args.gpu_ranks)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
        args.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        args.device_id = 0 if args.device == "cuda" else -1
        return args

    def extract(self, batch):
        exts, srctag = self.trainer.predict(batch)
        exts = [ext[0] for ext in exts]
        return exts, srctag

class SentExtractor:
    def __init__(self, ext_model_file,):
        import models.model_builder as model
        import models.trainer_ext as trainer_ext
        args = self._build_ext_args()
        checkpoint = torch.load(ext_model_file, map_location=lambda storage, loc: storage)
        self.name = 'BERT-Ext'
        self.model_file = ext_model_file
        self.model_ext = model.ExtSummarizer(args, args.device, checkpoint)
        self.model_ext.eval()
        self.decider = ExtDecider(logger)
        self.decider.load(ext_model_file + '.config')
        self.trainer = trainer_ext.build_trainer(args, args.device_id, self.model_ext, None)

    def _build_ext_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-task", default='ext')
        parser.add_argument("-encoder", default='bert')
        parser.add_argument("-mode", default='train')
        parser.add_argument("-bert_model_path", default='./bert_pretrained/')
        parser.add_argument("-bert_data_path", default='./bert_data/cnndm')
        parser.add_argument("-model_path", default='./models/')
        parser.add_argument("-result_path", default='./results/cnndm')
        parser.add_argument('-log_file', default='./logs/cnndm.log')
        parser.add_argument("-temp_dir", default='./temp')
        parser.add_argument("-train_from", default='')

        parser.add_argument("-max_pos", default=512, type=int)
        parser.add_argument("-max_tgt_len", default=140, type=int)
        parser.add_argument("-max_n_tags", default=6, type=int)
        parser.add_argument("-use_interval", default=True, type=bool)
        parser.add_argument("-large", default=False, type=bool)

        parser.add_argument("-ext_dropout", default=0.2, type=float)
        parser.add_argument("-ext_layers", default=2, type=int)
        parser.add_argument("-ext_hidden_size", default=768, type=int)
        parser.add_argument("-ext_heads", default=8, type=int)
        parser.add_argument("-ext_ff_size", default=2048, type=int)

        parser.add_argument("-param_init", default=0, type=float)
        parser.add_argument("-param_init_glorot", default=True, type=bool)
        parser.add_argument("-optim", default='adam')
        parser.add_argument("-lr", default=2e-3, type=float)
        parser.add_argument("-beta1", default=0.9, type=float)
        parser.add_argument("-beta2", default=0.999, type=float)
        parser.add_argument("-max_grad_norm", default=0, type=float)

        parser.add_argument("-train_steps", default=40000, type=int)
        parser.add_argument("-warmup_steps", default=10000, type=int)
        parser.add_argument("-report_every", default=50, type=int)
        parser.add_argument("-test_start_from", default=10000, type=int)
        parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
        parser.add_argument("-batch_size", default=8*512, type=int)
        parser.add_argument("-accum_count", default=2, type=int)
        parser.add_argument('-visible_gpus', default='0', type=str)
        parser.add_argument("-test_batch_size", default=8*512, type=int)

        parser.add_argument("-finetune_bert", default=True, type=bool)
        parser.add_argument('-gpu_ranks', default='0', type=str)
        parser.add_argument('-seed', default=666, type=int)

        parser.add_argument("-test_all", default=True, type=bool)
        parser.add_argument("-test_from", default='')
        parser.add_argument("-recall_eval", default=False, type=bool)
        parser.add_argument("-report_rouge", default=True, type=bool)
        parser.add_argument("-block_trigram", default=False, type=bool)

        args = parser.parse_args('')
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
        args.world_size = len(args.gpu_ranks)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
        args.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        args.device_id = 0 if args.device == "cuda" else -1
        return args

    def extract(self, batch):
        sel_scores, sel_ids = self.trainer.predict(batch)
        srcext = self.trainer.generate_srcext(batch, sel_scores, sel_ids, self.decider)
        srctag = self.trainer.generate_srctag(batch, sel_scores, sel_ids, self.decider)
        srcext = [ext[0] for ext in srcext]
        return srcext, srctag


class GuidAbsHandler:
    def __init__(self, model_file, data_path, result_path):
        self.data_path = data_path
        self.result_path = result_path
        self.model_file = model_file

    def _build_abs_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-task", default='abs')
        parser.add_argument("-encoder", default='bert')
        parser.add_argument("-mode", default='train')
        parser.add_argument("-bert_model_path", default='./bert_pretrained/')
        parser.add_argument("-bert_data_path", default='./bert_data/cnndm')
        parser.add_argument("-model_path", default='./models/')
        parser.add_argument("-result_path", default='./results/cnndm')
        parser.add_argument("-temp_dir", default='./temp')

        parser.add_argument("-train_steps", default=200000, type=int)
        parser.add_argument("-warmup_steps_bert", default=20000, type=int)
        parser.add_argument("-warmup_steps_dec", default=10000, type=int)
        parser.add_argument("-report_every", default=50, type=int)
        parser.add_argument("-save_checkpoint_steps", default=2000, type=int)
        parser.add_argument("-test_start_from", default=120000, type=int)
        parser.add_argument("-batch_size", default=560, type=int)
        parser.add_argument("-test_batch_size", default=3000, type=int)
        parser.add_argument("-accum_count", default=3, type=int)
        parser.add_argument('-visible_gpus', default='0', type=str)

        parser.add_argument("-max_pos", default=512, type=int)
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
        parser.add_argument('-log_file', default='../logs/cnndm.log')
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
        parser.add_argument("-word_dropout", default=0.3, type=float)
        parser.add_argument("-sent_dropout", default=0.2, type=float)

        args = parser.parse_args('')
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
        args.world_size = len(args.gpu_ranks)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

        args.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        args.device_id = 0 if args.device == "cuda" else -1
        return args

    def test_model(self, extractor, corpus_type='test', block_trigram=True, quick_test=False):
        logger.info('Test SentExt model (%s) and GuidAbs model (%s) ...' % (extractor.name, self.model_file))
        testname = '%s_guidabs_%s' % (extractor.name, 'blocktrigram' if block_trigram else 'noblocktrigram')
        # buid args
        args = self._build_abs_args()
        args.mode = 'test'
        args.bert_data_path = path.join(self.data_path, 'cnndm')
        args.model_path = self.result_path
        args.log_file = path.join(self.result_path, 'test_varextabs.%s.log' % testname)
        args.result_path = path.join(self.result_path, 'cnndm_' + testname)
        args.block_trigram = block_trigram
        init_logger(args.log_file)
        # load abs model
        abs_model_file = self.model_file
        logger.info('Loading abs model %s' % abs_model_file)
        step_abs = int(abs_model_file.split('.')[-2].split('_')[-1])
        checkpoint = torch.load(abs_model_file, map_location=lambda storage, loc: storage)
        model_abs = model_bld.AbsSummarizer(args, args.device, checkpoint)
        model_abs.eval()
        # init model testers
        tokenizer = BertTokenizer.from_pretrained(path.join(args.bert_model_path, model_abs.bert.model_name), do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

        predictor = pred_abs.build_predictor(args, tokenizer, symbols, model_abs, logger)
        test_iter = data_ldr.Dataloader(args, data_ldr.load_dataset(args, corpus_type, shuffle=False),
                                            args.test_batch_size, args.device,
                                            shuffle=False, is_test=True, keep_order=True)

        logger.info('Generating Ext/GuidAbs results %s ...' % args.result_path)
        avg_f1 = test_ext_abs(logger, args, extractor, predictor, 0, step_abs, test_iter, quick_test=quick_test)
        return avg_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # split bert data
    parser.add_argument("-source_data_path", default='./bert_data/')
    parser.add_argument("-result_path", default='./results/exp_varext_guidabs/')
    parser.add_argument("-log_file", default='./logs/exp_varext_guidabs.log')
    parser.add_argument("-quick_test", nargs='?', const=True, default=False, type=bool)
    args = parser.parse_args()

    if args.quick_test:
        logger.warning('WARNING: Running with -quick_test, which only tests 100 samples for verifying the code.')

    set_logger(logger, args.log_file)
    abs = GuidAbsHandler('./models/GuidAbs.model_step_222000.pt', args.source_data_path, args.result_path)

    logger.info("==== Apply to Lead3 extractor ====")
    lead3 = Lead3Extractor()
    abs.test_model(lead3, quick_test=args.quick_test)

    logger.info("==== Apply to BERTSUMEXT extractor ====")
    bertext = BertSumExtractor('./models/BERTSUMEXT.pt', block_trigram=False)
    abs.test_model(bertext, quick_test=args.quick_test)

    logger.info("==== Apply to BERT-Ext extractor ====")
    sentext = SentExtractor('./models/SentExt.model_step_25000.pt')
    abs.test_model(sentext, quick_test=args.quick_test)

