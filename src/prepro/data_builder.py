import gc
import glob
import argparse
import logging
import os
import os.path as path
import torch
from multiprocess import Pool, Manager
from rouge import Rouge
import numpy as np


logger = logging.getLogger()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


''' Algorithm for matching closest sentence in article for each summary sentence
'''
def match_by_rouge12(article, abstract):
    rouge = Rouge(metrics=["rouge-1", "rouge-2"])
    res = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        scores = rouge.get_scores(hyps, refs)
        recalls = [(score["rouge-1"]["r"] + score["rouge-2"]["r"]) / 2 for score in scores]
        res.append(recalls)
    return res

def match_by_rougeL(article, abstract):
    rouge = Rouge(metrics=["rouge-l"])
    res = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        scores = rouge.get_scores(hyps, refs)
        recalls = [score["rouge-l"]["r"] for score in scores]
        res.append(recalls)
    return res

def match_by_rouge12L(article, abstract):
    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    res = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        scores = rouge.get_scores(hyps, refs)
        recalls = [(score["rouge-1"]["r"] + score["rouge-2"]["r"] + score["rouge-l"]["r"]) / 3 for score in scores]
        res.append(recalls)
    return res

''' Extend current BERT data to include ROUGE-L recalls and oracle selection for each summay sentence
'''

def extend_to_guidabs(args):
    files = [fn for fn in glob.glob(args.bert_data_files)]
    a_lst = []
    for fn in files:
        real_name = path.basename(fn)
        save_file = path.join(args.result_data_path, real_name)
        if (os.path.exists(save_file)):
            logger.info('Exist and ignore %s' % save_file)
            continue

        jobs = torch.load(fn)
        a_lst.append((real_name, args, jobs, save_file))

    pool = Pool(args.n_cpus)
    for _ in pool.imap(_extend_to_guidabs, a_lst):
        pass

    pool.close()
    pool.join()

def _extend_to_guidabs(params, scorer=None):
    real_name, args, jobs, save_file = params

    logger.info('Processing %s' % real_name)
    datasets = []
    for d in jobs:
        # follow min_src_nsents/3: training, but not testing.
        # follow max_src_nsents/100: src_subtoken_idxs, sent_labels. src_txt is not truncated.
        # follow min_src_ntokens_per_sent/5: all fields, include src_text.
        # follow max_src_ntokens_per_sent/200: src_subtoken_idxs, cls_ids, segments_ids. src_txt is not truncated.
        # follow min_tgt_ntokens/5: tgt_subtoken_idxs for training, but not testing.
        # follow max_tgt_ntokens/500: tgt_subtoken_idxs
        src_subtoken_idxs = d["src"]
        tgt_subtoken_idxs = d["tgt"]
        sent_labels = d["src_sent_labels"]
        segments_ids = d["segs"]
        cls_ids = d["clss"]
        src_txt = d["src_txt"]
        tgt_txt = d["tgt_txt"].split('<q>')

        # make src_txt following max_src_nsents and max_src_ntokens_per_sent firstly
        src_txt = [' '.join(sent.split()[:args.max_src_ntokens_per_sent]) for sent in src_txt][:args.max_src_nsents]

        # verify consistency between data fields
        assert len(cls_ids) == len(src_txt), "len of cls_ids %s, num of source sentences %s" % (len(cls_ids), len(src_txt))
        unused_ids = [i for i, idx in enumerate(tgt_subtoken_idxs) if idx in [1, 3]]
        if len(unused_ids) != len(tgt_txt):
            logging.info("len of unused_ids %s, num of target sentences %s" % (len(unused_ids), len(tgt_txt)))
            tgt_txt = tgt_txt[:len(unused_ids)]

        # match oracle sentence for each summary
        try:
            # abs_art_scores = np.array(match_by_rougeL(src_txt, tgt_txt))
            abs_art_scores = np.array(match_by_rouge12L(src_txt, tgt_txt))
        except Exception as ex:
            # logger.warning("Ignore exception from match_by_rougeL: %s, len of src_txt %s, len of tgt_txt %s" % (ex, len(src_txt), len(tgt_txt)))
            logger.warning("Ignore exception from match_by_rouge12L: %s, len of src_txt %s, len of tgt_txt %s" % (ex, len(src_txt), len(tgt_txt)))
            continue

        abs_art_idx = np.argmax(abs_art_scores, axis=1)
        assert abs_art_scores.shape[0] == len(unused_ids)
        assert abs_art_scores.shape[1] == len(cls_ids)

        # generate guide tags for each summary sentence
        src_tags = np.zeros((len(src_subtoken_idxs), len(abs_art_idx)), dtype=np.int)
        for i, idx in enumerate(abs_art_idx):
            start = cls_ids[idx]
            end = cls_ids[idx + 1] if idx + 1 < len(cls_ids) else len(src_subtoken_idxs)
            src_tags[start:end, i] = 1
        src_tags = src_tags.tolist()

        tgt_tags = np.zeros(len(tgt_subtoken_idxs), dtype=np.int)
        for i in range(len(unused_ids)):
            start = unused_ids[i]
            end = unused_ids[i + 1] if i + 1 < len(unused_ids) else len(tgt_subtoken_idxs)
            tgt_tags[start:end] = i + 1  # 0 is skipped for padding
        tgt_tags = tgt_tags.tolist()

        data_dict = {"src": d["src"], "tgt": d["tgt"],
                       "src_sent_labels": d["src_sent_labels"], "segs": d["segs"], 'clss': d["clss"],
                       "src_txt": d["src_txt"], "tgt_txt": d["tgt_txt"],
                       "abs_art_idx": abs_art_idx, "src_tags": src_tags, "tgt_tags": tgt_tags}
        datasets.append(data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bert_data_files", default='./source_bert_data/cnndm.*0.bert.pt')
    parser.add_argument("-result_data_path", default='./bert_data/')
    parser.add_argument("-n_cpus", default=4, type=int)
    parser.add_argument("-temp_dir", default='./temp/')
    parser.add_argument('-log_file', default='./logs/data_builder.log')
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_file, format="[%(asctime)s %(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    # adding GuidAbs oracle info into existing BERT data
    extend_to_guidabs(args)

