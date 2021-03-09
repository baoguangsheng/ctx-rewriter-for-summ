#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import codecs
import struct

from exp_base import report_rouge
from others.logging import init_logger, logger
from tensorflow.core.example import example_pb2


def lead3_abisee(args):
    data_file = '/home2/baogs/pointer-generator-master/data/cnn-dailymail-master/finished_files/test.bin'
    data = []
    with open(data_file, 'rb') as fin:
        while True:
            len = fin.read(8)
            if not len:
                break
            len = struct.unpack('q', len)[0]
            exstr = struct.unpack('%ds' % len, fin.read(len))[0]
            tfex = example_pb2.Example()
            tfex.ParseFromString(exstr)
            article = tfex.features.feature['article'].bytes_list.value[0]
            abstract = tfex.features.feature['abstract'].bytes_list.value[0]
            article = article.decode().split(' <SEP> ')
            abstract = abstract.decode().split(' <SEP> ')
            data.append((article, abstract))

    gold_path = args.result_path + '.lead3.gold'
    can_path = args.result_path + '.lead3.candidate'
    gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
    can_out_file = codecs.open(can_path, 'w', 'utf-8')

    ct = 0
    for art, abs in data:
        gold_str = ' '.join(abs)
        pred_str = ' '.join(art[:3])

        can_out_file.write(pred_str + '\n')
        gold_out_file.write(gold_str + '\n')
        ct += 1

    can_out_file.close()
    gold_out_file.close()

    # calc rouge
    report_rouge(logger, args, gold_path, can_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-analysis", default='rouge', type=str, choices=['case_study', 'rouge', 'nyt', 'xsum', 'test_ext_abs'])
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-bert_data_path", default='../../shared/bert_data/bert_data_cnndm_final_rL/cnndm')
    parser.add_argument("-log_file", default='../models/baseline.log')
    parser.add_argument("-result_path", default='../models/cnndm')

    args = parser.parse_args()
    init_logger(args.log_file)

    lead3_abisee(args)

