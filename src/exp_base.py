import numpy as np
import json
import codecs
import torch
from tqdm import tqdm
from others.utils import rouge_results_to_str, test_rouge, avg_rouge_f1

class ExtDecider:
    def __init__(self, logger):
        self.logger = logger
        self.labels = []
        self.sel_scores = []
        self.sel_ids = []
        self.min_sel = None
        self.max_sel = None
        self.threshold = None

    def update(self, labels, sel_scores, sel_ids):
        self.labels.extend(labels.cpu().data.numpy().tolist())
        self.sel_scores.extend(sel_scores.cpu().data.numpy().tolist())
        self.sel_ids.extend(sel_ids.cpu().data.numpy().tolist())

    def finish(self):
        max_len = max([len(v) for v in self.labels])
        labels = np.array([v + [0]*(max_len - len(v)) for v in self.labels])
        sel_scores = np.array([v + [0] * (max_len - len(v)) for v in self.sel_scores])
        total_true = np.sum(labels)
        best_thresholds = []
        for min_sel in [1, 2, 3]:
            for max_sel in [3, 4, 5]:
                results = []
                for threshold in np.arange(0, 1, 0.01):
                    sel_pred = sel_scores > threshold
                    sel_pred[:, :min_sel] = 1
                    sel_pred[:, max_sel:] = 0
                    total_positive = np.sum(sel_pred)
                    true_positive = np.sum(labels * sel_pred)
                    precision = true_positive / total_positive
                    recall = true_positive / total_true
                    f1 = 2 * precision * recall / (precision + recall)
                    results.append((threshold, precision, recall, f1))
                results = sorted(results, key=lambda x: x[3], reverse=True)
                best_thresholds.append((min_sel, max_sel, results[0]))
                self.logger.info('min_sel %s max_sel %s best_threshold %s' % (min_sel, max_sel, results[0]))
        best_thresholds = sorted(best_thresholds, key=lambda x: x[2][3], reverse=True)
        self.min_sel = best_thresholds[0][0]
        self.max_sel = best_thresholds[0][1]
        self.threshold = best_thresholds[0][2][0]
        self.logger.info('Best: min_sel %s max_sel %s best_threshold %s' % (self.min_sel, self.max_sel, self.threshold))

    def save(self, to_file):
        obj = {'min_sel': self.min_sel, 'max_sel': self.max_sel, 'threshold': self.threshold}
        with open(to_file, 'w') as fout:
            json.dump(obj, fout)

    def load(self, from_file):
        with open(from_file, 'r') as fin:
            obj = json.load(fin)
        self.min_sel = obj['min_sel']
        self.max_sel = obj['max_sel']
        self.threshold = obj['threshold']

class Counter:
    def __init__(self, name):
        self.name = name
        self.min = None
        self.max = None
        self.avg = None
        self.sum = 0
        self.cnt = 0

    def count(self, val):
        self.sum += val
        self.cnt += 1
        self.avg = self.sum / self.cnt
        if self.min is None or self.min > val:
            self.min = val
        if self.max is None or self.max < val:
            self.max = val

    def __str__(self):
        return '%s: min %s, avg %s, max %s' % (self.name, self.min, self.avg, self.max)

class Extractor:
    def __init__(self, trainer, decider):
        self.trainer = trainer
        self.decider = decider

    def extract(self, batch):
        sel_scores, sel_ids = self.trainer.predict(batch)
        srcext = self.trainer.generate_srcext(batch, sel_scores, sel_ids, self.decider)
        srctag = self.trainer.generate_srctag(batch, sel_scores, sel_ids, self.decider)
        return srcext, srctag


def report_rouge(logger, args, gold_path, can_path):
    logger.info("Calculating Rouge")
    rouges = test_rouge(args.temp_dir, can_path, gold_path)
    logger.info('Rouges:\n%s' % rouge_results_to_str(rouges))
    avg_f1 = avg_rouge_f1(rouges)
    logger.info('Average Rouge F1: %s' % avg_f1)
    return avg_f1

def report_avglen(logger, args, gold_path, can_path):
    def _report_avglen(name, lines):
        words = Counter('%s: Avg Summary words' % name)
        sents = Counter('%s: Avg Summary sentences' % name)
        swords = Counter('%s: Avg Sentence words' % name)

        for line in lines:
            ss = [len(s.split()) for s in line.split('<q>')]
            sents.count(len(ss))
            words.count(sum(ss))
            for s in ss:
                swords.count(s)

        logger.info(sents)
        logger.info(words)
        logger.info(swords)

    references = [line.strip() for line in open(gold_path, encoding='utf-8')]
    _report_avglen('Ref', references)

    candidates = [line.strip() for line in open(can_path, encoding='utf-8')]
    _report_avglen('Gen', candidates)

def test_ext_abs(logger, args, extractor, predictor, step_ext, step_abs, test_iter, quick_test=False):
    gold_path = args.result_path + '.%s_%s.gold' % (step_ext, step_abs)
    canext_path = args.result_path + '.%s_%s.candidate_ext' % (step_ext, step_abs)
    canabs_path = args.result_path + '.%s_%s.candidate_abs' % (step_ext, step_abs)
    raw_src_path = args.result_path + '.%s_%s.raw_src' % (step_ext, step_abs)
    gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
    canext_out_file = codecs.open(canext_path, 'w', 'utf-8')
    canabs_out_file = codecs.open(canabs_path, 'w', 'utf-8')
    src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

    ct = 0
    logger.info('Files generating: %s, %s, %s, %s' % (canext_path, canabs_path, gold_path, raw_src_path))
    with torch.no_grad():
        for batch in tqdm(test_iter):
            exts, srctag = extractor.extract(batch)
            batch.tag_src = srctag
            batch_data = predictor.translate_batch(batch)
            translations = predictor.from_batch(batch_data)
            # translations = exts

            assert len(exts) == batch.batch_size
            assert len(exts) == len(translations)
            for ext, trans in zip(exts, translations):
                canext_out_file.write(ext.strip() + '\n')

                pred, gold, src = trans
                pred_str = pred.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
                gold_str = gold.strip()
                canabs_out_file.write(pred_str + '\n')
                gold_out_file.write(gold_str + '\n')
                src_out_file.write(src.strip() + '\n')
                ct += 1

            canext_out_file.flush()
            canabs_out_file.flush()
            gold_out_file.flush()
            src_out_file.flush()

            if quick_test and ct > 100:
                break

    canext_out_file.close()
    canabs_out_file.close()
    gold_out_file.close()
    src_out_file.close()
    logger.info('Files generated: %s, %s, %s, %s' % (canext_path, canabs_path, gold_path, raw_src_path))

    # calc rouge
    logger.info('Results for Ext:')
    report_avglen(logger, args, gold_path, canext_path)
    report_rouge(logger, args, gold_path, canext_path)

    logger.info('Results for ExtAbs:')
    report_avglen(logger, args, gold_path, canabs_path)
    avg_f1 = report_rouge(logger, args, gold_path, canabs_path)
    return avg_f1

def test_abs(logger, args, predictor, step_abs, test_iter):
    gold_path = args.result_path + '.%d.gold' % (step_abs)
    can_path = args.result_path + '.%d.candidate_abs' % (step_abs)
    raw_src_path = args.result_path + '.%d.raw_src' % (step_abs)
    gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
    can_out_file = codecs.open(can_path, 'w', 'utf-8')
    src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

    ct = 0
    logger.info('Files generating: %s, %s, %s' % (can_path, gold_path, raw_src_path))
    with torch.no_grad():
        for batch in tqdm(test_iter):
            batch_data = predictor.translate_batch(batch)
            translations = predictor.from_batch(batch_data)

            for trans in translations:
                pred, gold, src = trans
                pred_str = pred.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
                gold_str = gold.strip()

                can_out_file.write(pred_str + '\n')
                gold_out_file.write(gold_str + '\n')
                src_out_file.write(src.strip() + '\n')
                ct += 1

            can_out_file.flush()
            gold_out_file.flush()
            src_out_file.flush()

    can_out_file.close()
    gold_out_file.close()
    src_out_file.close()
    logger.info('Files generated: %s, %s, %s' % (can_path, gold_path, raw_src_path))

    # calc rouge
    report_avglen(logger, args, gold_path, can_path)
    avg_f1 = report_rouge(logger, args, gold_path, can_path)
    return avg_f1

def test_ext(logger, args, trainer, step_ext, test_iter, decider):
    gold_path = args.result_path + '.%d.gold' % step_ext
    can_path = args.result_path + '.%d.candidate_ext' % step_ext
    raw_src_path = args.result_path + '.%d.raw_src' % step_ext
    gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
    can_out_file = codecs.open(can_path, 'w', 'utf-8')
    src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

    ct = 0
    logger.info('Files generating: %s, %s, %s' % (can_path, gold_path, raw_src_path))
    with torch.no_grad():
        for batch in tqdm(test_iter):
            sel_scores, sel_ids = trainer.predict(batch)
            translations = trainer.generate_srcext(batch, sel_scores, sel_ids, decider)

            for trans in translations:
                pred, gold, src = trans
                can_out_file.write(pred.strip() + '\n')
                gold_out_file.write(gold.strip() + '\n')
                src_out_file.write(src.strip() + '\n')
                ct += 1

            can_out_file.flush()
            gold_out_file.flush()
            src_out_file.flush()

    can_out_file.close()
    gold_out_file.close()
    src_out_file.close()
    logger.info('Files generated: %s, %s, %s' % (can_path, gold_path, raw_src_path))

    report_avglen(logger, args, gold_path, can_path)
    avg_f1 = report_rouge(logger, args, gold_path, can_path)
    return avg_f1


if __name__ == '__main__':
    import argparse
    from others.logging import init_logger, logger
    parser = argparse.ArgumentParser()
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-log_file", default='../temp/exp_base.log')
    parser.add_argument("-task", default='rouge')
    parser.add_argument("-gold", default='../exp_main_sentext_guidabs_r12L/models_guidabs_maxpos512_noproj_worddrop0.3_sentdrop0.2/cnndm.BERTSUMEXT_noblocktrigram_222000.gold')
    parser.add_argument("-candi", default='../exp_main_sentext_guidabs_r12L/models_guidabs_maxpos512_noproj_worddrop0.3_sentdrop0.2/cnndm.BERTSUMEXT_noblocktrigram_222000.candidate_ext')
    args = parser.parse_args()

    init_logger(args.log_file)

    if args.task == 'rouge':
        report_rouge(logger, args, args.gold, args.candi)
    elif args.task == 'avglen':
        report_avglen(logger, args, args.gold, args.candi)
