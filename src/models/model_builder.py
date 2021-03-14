import copy
import os.path as path
import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder, MatchTransformerEncoder
from models.optimizers import Optimizer

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(args, vocab_size, dec_hidden_size, gen_weight=None):
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        nn.LogSoftmax(dim=-1)
    )
    if gen_weight is not None:
        generator[0].weight = gen_weight

    return generator

class Bert(nn.Module):
    def __init__(self, bert_model_path, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        self.model_name = 'bert-large-uncased' if large else 'bert-base-uncased'
        self.model = BertModel.from_pretrained(path.join(bert_model_path, self.model_name), cache_dir=temp_dir)
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.bert_model_path, args.large, args.temp_dir, args.finetune_bert)
        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        segs = (1 - segs % 2) * mask_src.long()
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls

class TiedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(TiedEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx,
             max_norm, norm_type, scale_grad_by_freq, sparse, _weight)

    def forward(self, input):
        return super(TiedEmbedding, self).forward(input)

    def matmul(self, tags):
        return tags.matmul(self.weight[1:1 + tags.size(2)])

class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.bert_model_path, args.large, args.temp_dir, args.finetune_bert)

        max_pos = args.max_pos
        if(max_pos>512):
            my_pos_embeddings = nn.Embedding(max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        # guide-tags
        self.tag_embeddings = TiedEmbedding(args.max_n_tags, self.bert.model.config.hidden_size, padding_idx=0)
        self.tag_drop = nn.Dropout(args.tag_dropout)

        # decoder
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.bert.model.config.hidden_size, heads=self.args.dec_heads, d_ff=self.args.dec_ff_size,
            dropout=self.args.dec_dropout, embeddings=tgt_embeddings, tag_embeddings=self.tag_embeddings)

        # generator
        self.generator = get_generator(args, self.vocab_size, self.bert.model.config.hidden_size, gen_weight=self.decoder.embeddings.weight)

        # load checkpoint or initialize the parameters
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            self.tag_embeddings.weight.data.normal_(mean=0.0, std=0.02)
            self.tag_embeddings.weight[self.tag_embeddings.padding_idx].data.fill_(0)
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                self.decoder.embeddings.weight.data.copy_(self.bert.model.embeddings.word_embeddings.weight)

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls, tag_src, tag_tgt):
        segs_src = (1 - segs % 2) * mask_src.long()
        top_vec = self.bert(src, segs_src, mask_src)
        if self.training and self.args.sent_dropout > 0:
            idx = (torch.arange(clss.size(1), device=clss.device) + 1).unsqueeze(0).expand_as(clss)  # n x sents
            drop = torch.rand(clss.size(), dtype=torch.float, device=clss.device) < self.args.sent_dropout  # n x sents
            idx = idx * drop.long()
            msk_drop = torch.sum((segs.unsqueeze(-2) == idx.unsqueeze(-1)).float(), dim=1)  # n x 512
            msk_tag = (torch.sum(tag_src, dim=2) > 0).float()  # n x 512
            msk_drop = msk_drop * (1 - msk_tag) * mask_src.float()
            top_vec = top_vec * (1 - msk_drop).unsqueeze(-1)
        tag_vec = self.tag_embeddings.matmul(tag_src)
        top_vec = top_vec + self.tag_drop(tag_vec)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        if self.training and self.args.word_dropout > 0:
            word_mask = 103
            drop = torch.rand(tgt.size(), dtype=torch.float, device=tgt.device) < self.args.word_dropout
            drop = drop * mask_tgt
            tgt = torch.where(drop, tgt.new_full(tgt.size(), word_mask), tgt)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state, tag=tag_tgt[:, :-1])
        return decoder_outputs, None
