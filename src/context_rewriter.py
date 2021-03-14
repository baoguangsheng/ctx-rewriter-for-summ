import numpy as np
import json
import os
import codecs
import torch
import os.path as path
import logging
from tqdm import tqdm
from others.tokenization import BertTokenizer
import models.model_builder as model_bld
import models.predictor as pred_abs

logger = logging.getLogger(__name__)

''' Take document and extractive summary as input, generate abstractive summary.
'''
class Batch(object):
    def __init__(self, src, segs, tags, device=None):
        """Create a Batch from a list of examples."""
        self.batch_size = 1
        src = torch.tensor([src])
        segs = torch.tensor([segs])
        mask_src = 1 - (src == 0)
        tag_src = torch.tensor([tags], dtype=torch.float)

        setattr(self, 'src', src.to(device))
        setattr(self, 'segs', segs.to(device))
        setattr(self, 'mask_src', mask_src.to(device))
        setattr(self, 'tag_src', tag_src.to(device))

    def __len__(self):
        return self.batch_size

class ContextRewriter:
    def __init__(self, abs_model_file):
        self.args = self._build_abs_args()
        # load model
        step_abs = int(abs_model_file.split('.')[-2].split('_')[-1])
        checkpoint = torch.load(abs_model_file, map_location=lambda storage, loc: storage)
        self.model_abs = model_bld.AbsSummarizer(self.args, self.args.device, checkpoint)
        self.model_abs.eval()
        # prepare tokenizer and predictor
        self.tokenizer = BertTokenizer.from_pretrained(path.join(self.args.bert_model_path, self.model_abs.bert.model_name), do_lower_case=True)
        self.symbols = {'BOS': self.tokenizer.vocab['[unused0]'], 'EOS': self.tokenizer.vocab['[unused1]'],
                   'PAD': self.tokenizer.vocab['[PAD]'], 'EOQ': self.tokenizer.vocab['[unused2]']}
        self.predictor = pred_abs.build_predictor(self.args, self.tokenizer, self.symbols, self.model_abs, logger)
        # special tokens
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

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

        parser.add_argument("-train_steps", default=240000, type=int)
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
        parser.add_argument('-log_file', default='./logs/cnndm.log')
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
        parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
        parser.add_argument('-max_src_nsents', default=100, type=int)

        args = parser.parse_args('')
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
        args.world_size = len(args.gpu_ranks)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

        args.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        args.device_id = 0 if args.device == "cuda" else -1
        return args

    def _build_batch(self, doc_lines, ext_lines):
        # Same logic as BertData to process raw input
        src = [line.split()[:self.args.max_src_ntokens_per_sent] for line in doc_lines[:self.args.max_src_nsents]]
        src_txt = [' '.join(sent) for sent in src]

        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            segments_ids += s * [i+1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        # Generate group tag
        abs_art_idx = [doc_lines.index(line) for line in ext_lines]

        src_tags = np.zeros((len(src_subtoken_idxs), len(abs_art_idx)), dtype=np.int)
        for i, idx in enumerate(abs_art_idx):
            start = cls_ids[idx]
            end = cls_ids[idx + 1] if idx + 1 < len(cls_ids) else len(src_subtoken_idxs)
            src_tags[start:end, i] = 1
        assert np.sum(src_tags[self.args.max_pos:]) == 0, 'Extractive position out of 512 tokens.'
        src_tags = src_tags.tolist()
        # Generate batch
        src_subtoken_idxs = src_subtoken_idxs[:self.args.max_pos]
        segments_ids = segments_ids[:self.args.max_pos]
        src_tags = src_tags[:self.args.max_pos]
        return Batch(src_subtoken_idxs, segments_ids, src_tags, device=self.args.device)

    def rewrite(self, doc_lines, ext_lines):
        assert all([line in doc_lines for line in ext_lines])

        try:
            batch = self._build_batch(doc_lines, ext_lines)
            batch_data = self.predictor.translate_batch(batch)
            preds = batch_data["predictions"]
            assert len(preds) == 1
            pred_lines = self.tokenizer.convert_ids_to_tokens([int(n) for n in preds[0][0]])
            pred_lines = ' '.join(pred_lines).replace(' ##', '').replace('[unused0]', '').replace('[unused3]', '')\
                .replace('[PAD]', '').replace('[unused1]', '')\
                .replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
            pred_lines = pred_lines.split('<q>')
        except Exception as ex:
            print(ex)
            return ext_lines
        return pred_lines


if __name__ == '__main__':
    import argparse
    from others.logging import init_logger, logger
    parser = argparse.ArgumentParser()
    parser.add_argument("-log_file", default='./logs/exp_base.log')
    parser.add_argument("-model_file", default='./models/GuidAbs.model_step_222000.pt')
    args = parser.parse_args()

    init_logger(args.log_file)

    rewriter = ContextRewriter(args.model_file)

    doc_lines = ["georgia high school basketball coach greg scott had n't been feeling well lately , so on monday he went to his doctor where he was shocked to discover he had leukemia .",
                "less than 24 hours after the devastating diagnosis , the 51-year-old married father of two and grandfather unexpectedly succumbed to the deadly disease .",
                "for the past eight years , scott had taught special education and social studies at cass high school and was the school 's head basketball coach .",
                "sudden death : less than 24 hours after georgia high school basketball coach greg scott ( left ) was diagnosed with leukemia , he succumbed to the illness",
                "mentor : for the past eight years , scott ( pictured right with his son ) had taught special education and social studies at cass high school and was the school 's head basketball coach",
                "towering figure : the teacher and sports coach was universally remembered as a deeply caring , wise man who was committed to his students and players",
                "but those who knew him , including his colleagues and players , say there was much more to scott than his love of sports .",
                "false hope : initially , scott ( pictured with daughter cieanna ) and his family were told that his leukemia was treatable",
                "greg scott , 51 , a basketball coach at cass high school in georgia , likely had contracted a fast-moving type of leukemia , known as acute myeloid leukemia , months before his death but was diagnosed only a day before he lost his brief battle with the illness .",
                "leukemia is a type of cancer that starts in cells that form new blood cells .",
                "these cells are found in the soft , inner part of the bones called the bone marrow .",
                "in patients who suffers from aml , the cancer grows quickly , and if not treated , could be fatal in a matter of months .",
                "with acute types of leukemia such as aml , bone marrow cells do not mature the way they 're supposed to .",
                "these immature cells , called blast cells , just keep building up .",
                "because it is ` acute , ' this type of leukemia can spread quickly to the blood and to other parts of the body such as lymph nodes ; liver ; spleen ; brain and spinal cord , as well as testicles in men .",
                "this type of cancer is considered rare and has a five-year survival rate of only 24 per cent ."]
    ext_lines = ["georgia high school basketball coach greg scott had n't been feeling well lately , so on monday he went to his doctor where he was shocked to discover he had leukemia .",
                "less than 24 hours after the devastating diagnosis , the 51-year-old married father of two and grandfather unexpectedly succumbed to the deadly disease .",
                "for the past eight years , scott had taught special education and social studies at cass high school and was the school 's head basketball coach ."]

    res_lines = rewriter.rewrite(doc_lines, ext_lines)

    print('Extractive Summary:')
    for line in ext_lines:
        print('\t', line)
    print('Rewritten Summary:')
    for line in res_lines:
        print('\t', line)

