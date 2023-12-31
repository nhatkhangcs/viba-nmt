import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from model.transformerbertpgn.NoamLRScheduler import *
from model.transformerbertpgn.utils import process_batch, format, no_accent_vietnamese
from numpy.random import  uniform
import re
from nltk.translate.bleu_score import corpus_bleu

class NMT(pl.LightningModule):
    def __init__(self, dictionary, tokenizer, annotator, criterion, d_model, d_ff, num_heads, num_layers, 
                dropout, bart=None, d_bart=None, use_pgn=False, use_ner=False, 
                max_src_len=256, max_tgt_len=256): #TODO separate bart to pe and pd
        super(NMT, self).__init__()
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.annotator = annotator
        self.criterion = criterion
        self.model = Transformer(
            vocab_size=len(self.dictionary), 
            d_model=d_model,
            d_ff=d_ff, 
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            bart=bart,
            d_bart=d_bart, 
            padding_idx=self.dictionary.token_to_index(self.dictionary.pad_token), 
            unk_idx=self.dictionary.token_to_index(self.dictionary.unk_token),
            use_pgn=use_pgn,
            use_ner=use_ner
        )
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def training_step(self, batch, batch_idx):
        input = process_batch(
            batch, self.dictionary, self.tokenizer, self.annotator, 
            max_src_len=self.max_src_len, use_pgn=self.model.use_pgn, 
            use_ner=self.model.use_ner, device=self.device
        )

        output = self.model(
            input['src'], input['tgt'][:,:-1], 
            input['src_bart'], input['tgt_bart'][:,:-1], #TODO seperate src_bart to pe_input and pd_input
            input['src_ext'], input['src_ne'], input['max_oov_len']
        )  # [batch_size, seq_len, vocab_size] 

        tgt = input['tgt_ext'] if self.model.use_pgn else input['tgt']
        loss = self.criterion(output, tgt[:,1:]) 
        
        # log
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            should_log = (
                (self.global_step + 1) % self.trainer.log_every_n_steps == 0
            )
            if should_log:
                self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input = process_batch(
            batch, self.dictionary, self.tokenizer, self.annotator, 
            max_src_len=self.max_src_len, use_pgn=self.model.use_pgn, 
            use_ner=self.model.use_ner, device=self.device
        )
        
        output = self.model(
            input['src'], input['tgt'][:,:-1], 
            input['src_bart'], input['tgt_bart'][:,:-1], #TODO seperate src_bart to pe_input and pd_input
            input['src_ext'], input['src_ne'], input['max_oov_len']
        )  # [batch_size, seq_len, vocab_size]

        tgt = input['tgt_ext'] if self.model.use_pgn else input['tgt']
        loss = self.criterion(output, tgt[:,1:]) 

        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_val_loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        input = process_batch(
            batch, self.dictionary, self.tokenizer, self.annotator, 
            max_src_len=self.max_src_len, use_pgn=self.model.use_pgn, 
            use_ner=self.model.use_ner, device=self.device
        )

        preds = self.model.inference(
            input['src'], input['src_bart'], #TODO seperate src_bart to pe_input and pd_input
            self.dictionary, self.tokenizer,
            input['src_ext'], input['src_ne'], input['max_oov_len'], 
            self.max_tgt_len, self.dictionary.token_to_index(self.dictionary.eos_token)
        )

        # decode
        preds = preds.tolist()
        sequences = []
        decode_dict = input['dictionary_ext'] if self.model.use_pgn else self.dictionary
        for seq_ids in preds:
            tokens = [decode_dict.index_to_token(i) for i in seq_ids]
            seq = self.tokenizer.convert_tokens_to_string(tokens)
            sequences.append(self._postprocess(seq))

        # print results
        batch_size = self.trainer.test_dataloaders[0].batch_size
        for offset in range(len(input['src_raw'])):
            print(f'--|S-{batch_idx*batch_size + offset}: {input["src_raw"][offset]}')
            print(f'--|T-{batch_idx*batch_size + offset}: {input["tgt_raw"][offset]}')
            print(f'--|P-{batch_idx*batch_size + offset}: {sequences[offset]}')
            print()

        # compute bleu
        candidates = [seq.strip().split() for seq in sequences]
        references = [[ref.strip().split()] for ref in input['tgt_raw']]
        bleu = corpus_bleu(references, candidates)
        return {"bleu": bleu}

    def test_epoch_end(self, outputs):
        avg_bleu = torch.tensor([x['bleu'] for x in outputs], dtype=torch.float32).mean()
        self.log('bleu', avg_bleu, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = NoamLRScheduler(optimizer, warmup_steps=4000, d_model=512)
        lr_scheduler_config = {
            'scheduler': lr_scheduler,
            'interval': 'step'
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def _postprocess(self, seq):
        eos_idx = seq.find(self.dictionary.eos_token)
        if eos_idx != -1:
            seq = seq[:eos_idx]
        seq = re.sub(f"^{self.dictionary.bos_token}| {self.dictionary.bos_token}", "", seq)
        seq = re.sub(f"^{self.dictionary.pad_token}| {self.dictionary.pad_token}", "", seq)
        seq = re.sub(f"^{self.dictionary.unk_token}| {self.dictionary.unk_token}", "", seq)
        seq = no_accent_vietnamese(seq)
        return format(seq)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers, dropout, 
                bart=None, d_bart=None, #TODO separate bart to pe and pd
                padding_idx=None, unk_idx=None, use_pgn=False, use_ner=False):
        super(Transformer, self).__init__()
        if bart is None:
            assert d_bart is None
        self.src_embedding = Embedding(vocab_size=vocab_size, d_model=d_model, dropout=dropout, padding_idx=padding_idx)
        self.tgt_embedding = Embedding(vocab_size=vocab_size, d_model=d_model, dropout=dropout, padding_idx=padding_idx)
        self.encoder = Encoder(d_model=d_model, d_ff=d_ff, num_heads=num_heads, num_layers=num_layers, dropout=dropout, d_pe=d_bart)
        self.decoder = Decoder(d_model=d_model, d_ff=d_ff, num_heads=num_heads, num_layers=num_layers, dropout=dropout, d_pe=d_bart, d_pd=d_bart)
        self.linear = Linear(d_model, vocab_size)
        self.bart = bart
        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.use_pgn = use_pgn
        self.use_ner = use_ner
        self.pointer_generator = PointerGeneratorNetwork(d_model=d_model, dropout=dropout) if use_pgn else None

    def forward(self, src, tgt, src_bart, tgt_bart, src_ext=None, src_ne=None, max_oov_len=None):
        """
        Arguments:
            src: [batch_size, src_seq_len] mandatory: index of source tokens
            tgt: [batch_size, tgt_seq_len] mandatory: index of target tokens
            pe_input: [batch_size, src_seq_len] mandatory: index of source pe tokens
            src_ext: [batch_size, src_seq_len] optional: index of source tokens in extend vocab
            src_ne: [batch_size, src_seq_len] optional: name entity mask (1s for name entities, 0s otherwise)
            max_oov_len: int optional: number of oov in current batch
        Return:
            softmax distribution
        """
        if self.use_pgn:
            assert (src_ext is not None) and (max_oov_len is not None)
            assert src_ext.size() == src.size()
            assert max_oov_len >= 0

        if self.use_ner:
            assert src_ne is not None
            assert src_ne.size() == src.size()

        src_padding_mask = src.eq(self.padding_idx).unsqueeze(1) if self.padding_idx is not None else None
        tgt_padding_mask = tgt.eq(self.padding_idx).unsqueeze(1) if self.padding_idx is not None else None
        special_token_mask = self._special_token_mask(
            unk_mask=src.eq(self.unk_idx).unsqueeze(1) if self.unk_idx is not None else None,
            ne_mask=src_ne.unsqueeze(1) if self.use_ner else None
        )

        src_embedding = self.src_embedding(src)
        pe_out = self.bart(input_ids=src_bart).encoder_last_hidden_state.detach() if self.bart is not None else None
        pd_out = self.bart(input_ids=src_bart, decoder_input_ids=tgt_bart).last_hidden_state.detach() if self.bart is not None else None
        encoder_out = self.encoder(src_embedding, pe_out, padding_mask=src_padding_mask)

        tgt_embedding = self.tgt_embedding(tgt)
        decoder_out = self.decoder(
            tgt_embedding, encoder_out, pe_out, pd_out,
            src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask
        )

        vocab_dist = F.softmax(self.linear(decoder_out), dim=-1)

        final_dist = self.pointer_generator(
            encoder_out=encoder_out, 
            decoder_out=decoder_out, 
            vocab_dist=vocab_dist,
            decoder_in=tgt_embedding, 
            src_ext=src_ext, 
            max_oov_len=max_oov_len, 
            padding_mask=src_padding_mask, 
            special_token_mask=special_token_mask
        ) if self.use_pgn else vocab_dist

        return final_dist

    def inference(self, src, src_bart, dictionary, tokenizer, #TODO separate src_bart to pe_input and pd_input
                src_ext=None, src_ne=None, max_oov_len=None, max_tgt_len=256, eos_idx=None):
        """
        Arguments:
            src: [batch_size, src_seq_len]
            pe_input: [batch_size, src_seq_len]
            src_ext: [batch_size, src_seq_len]
            max_oov_len: [batch_size, src_seq_len]
        Return:
            preds: [batch_size, tgt_seq_len, vocab_size]
        """
        if self.use_pgn:
            assert (src_ext is not None) and (max_oov_len is not None)
            assert src_ext.size() == src.size()
            assert max_oov_len >= 0

        src_padding_mask = src.eq(self.padding_idx).unsqueeze(1) if self.padding_idx is not None else None
        special_token_mask = self._special_token_mask(
            unk_mask=src.eq(self.unk_idx).unsqueeze(1) if self.unk_idx is not None else None,
            ne_mask=src_ne.unsqueeze(1) if self.use_ner else None
        )

        src_embedding = self.src_embedding(src)
        pe_out = self.bart(input_ids=src_bart).encoder_last_hidden_state.detach() if self.bart is not None else None
        encoder_out = self.encoder(src_embedding, pe_out, padding_mask=src_padding_mask)

        preds = torch.tensor([0], device=src.device).repeat(src.size(0), 1) # [batch_size, current_len]
        tgt = preds.detach().clone()
        for _ in range(max_tgt_len - 1):
            tgt_embedding = self.tgt_embedding(tgt)
            tgt_padding_mask = tgt.eq(self.padding_idx).unsqueeze(1) if self.padding_idx is not None else None
            tgt_bart = preds[:,1:].detach().clone().to(device='cpu')
            tgt_bart.apply_(lambda x: tokenizer._convert_token_to_id(dictionary.index_to_token(x)))
            tgt_bart = torch.cat((torch.tensor([tokenizer.eos_token_id], device=src.device).repeat(tgt_bart.size(0), 1), tgt_bart.to(device=src.device)), dim=1)
            pd_out = self.bart(input_ids=src_bart, decoder_input_ids=tgt_bart).last_hidden_state.detach() if self.bart is not None else None
            decoder_out = self.decoder(
                tgt_embedding, encoder_out, pe_out, pd_out,
                src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask
            )

            vocab_dist = F.softmax(self.linear(decoder_out), dim=-1)

            final_dist = self.pointer_generator(
                encoder_out=encoder_out, 
                decoder_out=decoder_out, 
                vocab_dist=vocab_dist,
                decoder_in=tgt_embedding, 
                src_ext=src_ext, 
                max_oov_len=max_oov_len, 
                padding_mask=src_padding_mask, 
                special_token_mask=special_token_mask
            ) if self.use_pgn else vocab_dist # [batch_size, tgt_seq_len, vocab_size]
            
            values, ids = torch.max(final_dist, dim=-1)
            preds = torch.cat([preds,ids[:,-1].unsqueeze(1)],dim=-1)
            # map oov to unk before feed to next timestep
            vocab_size = self.tgt_embedding.embedding.num_embeddings
            tgt = torch.where(
                preds < vocab_size, 
                preds, 
                self.unk_idx
            )

            if eos_idx is not None:
                early_stopping = torch.all(preds.eq(eos_idx).sum(dim=-1)).item()
                if early_stopping:
                    break
        return preds

    def _special_token_mask(self, unk_mask=None, ne_mask=None):
        """
        Arguments:
            unk_mask: [batch_size, 1, seq_len] optional
            ne_mask: [batch_size, 1, seq_len] optional
        Return:
            special_token_mask: [batch_size, 1, seq_len] or None
        """
        if unk_mask is None:
            return ne_mask
        else:
            return torch.logical_or(unk_mask, ne_mask) if ne_mask is not None else unk_mask

class PointerGeneratorNetwork(nn.Module):
    def __init__(self, d_model, dropout):
        super(PointerGeneratorNetwork, self).__init__()
        self.encoder_decoder_attention = Attention(d_Q_in=d_model, d_K_in=d_model, d_V_in=d_model, d_k=d_model, d_v=d_model, dropout=dropout)
        self.context_to_pgen = Linear(d_model, 1)
        self.decoder_out_to_pgen = Linear(d_model, 1)
        self.decoder_in_to_pgen = Linear(d_model, 1)

    def forward(self, encoder_out, decoder_out, vocab_dist, decoder_in, src_ext, max_oov_len, padding_mask=None, special_token_mask=None):
        """
        :param mandatory Tensor[batch_size, src_seq_len, d_model] encoder_out
        :param mandatory Tensor[batch_size, tgt_seq_len, d_model] decoder_out
        :param mandatory Tensor[batch_size, tgt_seq_len, d_model] vocab_dist
        :param mandatory Tensor[batch_size, tgt_seq_len, d_model] decoder_in
        :param mandatory Tensor[batch_size, src_seq_len] src_ext: ids of tokens in extend dictionary
        :param optional Tensor[batch_size, 1, src_seq_len] padding_mask: prevent attention with padding token in src
        :param optional Tensor[batch_size, 1, src_seq_len] special_token_mask: prevent attention with special tokens in src
        """
        context, att_dist = self.encoder_decoder_attention(
            decoder_out, encoder_out, encoder_out, padding_mask=padding_mask, special_token_mask=special_token_mask
        ) # [batch_size, tgt_seq_len, d_model], [batch_size, tgt_seq_len, src_seq_len]

        extra_zeros = torch.zeros(vocab_dist.size(0), vocab_dist.size(1), max_oov_len, device=vocab_dist.device) # [batch_size, tgt_seq_len, max_oov_len]
        extend_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1) # [batch_size, tgt_seq_len, extend_vocab_size], extend_vocab_size = vocab_size + max_oov_len

        p_gen = torch.sigmoid(
            self.context_to_pgen(context) + self.decoder_out_to_pgen(decoder_out) + self.decoder_in_to_pgen(decoder_in)
        ) # [batch_size, tgt_seq_len, 1]

        # combine distribution
        extend_vocab_dist = p_gen * extend_vocab_dist # [batch_size, tgt_seq_len, extend_vocab_size]
        att_dist = (1.0 - p_gen) * att_dist # [batch_size, tgt_seq_len, src_seq_len]

        final_dist = extend_vocab_dist.scatter_add(
            dim=-1,
            index=src_ext.unsqueeze(1).repeat(1, extend_vocab_dist.size(1), 1), # [batch_size, tgt_seq_len,  src_seq_len]
            src=att_dist
        ) # [batch_size, tgt_seq_len, extend_vocab_size]

        return final_dist


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout, d_pe=None):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout, d_pe=d_pe)
            for _ in range(num_layers)
        ])

    def forward(self, x, pe_out=None, padding_mask=None):
        for layer in self.encoder_layers:
            x = layer(x, pe_out, padding_mask=padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout, d_pe=None, d_pd=None):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout, d_pe=d_pe, d_pd=d_pd)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_out, pe_out=None, pd_out=None, src_padding_mask=None, tgt_padding_mask=None):
        """
        :param mandatory Tensor[batch_size, tgt_seq_len, d_model] x: embedding of previous decoder layer
        :param mandatory Tensor[batch_size, src_seq_len, d_model] encoder_out: embedding of final encoder layer
        :param mandatory Tensor[batch_size, src_seq_len, d_pe] pe_out: embedding of pe
        :param optional Tensor[batch_size, 1, src_seq_len] src_padding_mask: padding mask for key from decoder
        :param optional Tensor[batch_size, 1, tgt_seq_len] tgt_padding_mask: padding mask for key from encoder, pe
        """
        for layer in self.decoder_layers:
            x = layer(
                x, encoder_out, pe_out, pd_out,
                src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask
            )
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, d_pe=None, 
        encoder_pe_dropout=True, encoder_pe_mixup=False, encoder_pe_dropout_ratio=0.5):
        """
        :param mandatory int d_model
        :param mandatory int d_ff
        """
        super(EncoderLayer, self).__init__()
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        assert d_k * num_heads == d_model and d_v * num_heads == d_model
        self.self_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_model, d_V_in=d_model, d_k=d_k, d_v=d_v, dropout=dropout)
        self.pe_enc_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_pe, d_V_in=d_pe, d_k=d_k, d_v=d_v, dropout=dropout) if d_pe is not None else None
        self.attention_residual = Residual(d_model=d_model,dropout=dropout)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.feed_forward_residual = Residual(d_model=d_model,dropout=dropout)
        
        self.encoder_pe_dropout = encoder_pe_dropout
        self.encoder_pe_mixup = encoder_pe_mixup
        self.encoder_pe_dropout_ratio = encoder_pe_dropout_ratio
        assert self.encoder_pe_dropout_ratio >= 0. and self.encoder_pe_dropout_ratio <= 0.5


    def forward(self, x, pe_out=None, padding_mask=None):
        """
        :param Tensor[batch_size, seq_len] x
        :return Tensor[batch_size, seq_len, d_model] position_encoding
        """
        self_attention = self.self_attention(x, x, x, padding_mask=padding_mask)
        pe_enc_attention = self.pe_enc_attention(x, pe_out, pe_out, padding_mask=padding_mask) if self.pe_enc_attention is not None else None
        ratios = self.get_ratios()
        pefused_attention = self.attention_residual(
            ratios[0]*self_attention + ratios[1]*pe_enc_attention if pe_enc_attention is not None else self_attention,
            x
        )

        return self.feed_forward_residual(
            self.feed_forward(pefused_attention), 
            pefused_attention
        )

    def get_ratios(self):
        if self.encoder_pe_dropout:
            frand = float(uniform(0, 1))
            if self.encoder_pe_mixup and self.training:
                return [frand, 1 - frand]
            if frand < self.encoder_pe_dropout_ratio and self.training:
                return [1, 0]
            elif frand > 1 - self.encoder_pe_dropout_ratio and self.training:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [0.5, 0.5]


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, d_pe=None, d_pd=None,
        decoder_pe_dropout=True, decoder_pe_mixup=False, decoder_pe_dropout_ratio=0.5, 
        decoder_pd_dropout=True, decoder_pd_mixup=False, decoder_pd_dropout_ratio=0.5):
        """
        :param mandatory int d_model
        :param mandatory int d_ff
        """
        super(DecoderLayer, self).__init__()
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        assert d_k * num_heads == d_model and d_v * num_heads == d_model
        self.self_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_model, d_V_in=d_model, d_k=d_k, d_v=d_v, dropout=dropout, masking=True)
        self.pd_dec_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_pd, d_V_in=d_pd, d_k=d_k, d_v=d_v, dropout=dropout, masking=True)
        self.pdfused_attention_residual = Residual(d_model=d_model,dropout=dropout)
        self.encoder_decoder_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_model, d_V_in=d_model, d_k=d_k, d_v=d_v, dropout=dropout)
        self.pe_dec_attention = MultiHeadAttention(num_heads=num_heads, d_Q_in=d_model, d_K_in=d_pe, d_V_in=d_pe, d_k=d_k, d_v=d_v, dropout=dropout) if d_pe is not None else None
        self.pefused_attention_residual = Residual(d_model=d_model,dropout=dropout)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.feed_forward_residual = Residual(d_model=d_model,dropout=dropout)

        self.decoder_pe_dropout = decoder_pe_dropout
        self.decoder_pe_mixup = decoder_pe_mixup
        self.decoder_pe_dropout_ratio = decoder_pe_dropout_ratio
        assert self.decoder_pe_dropout_ratio >= 0. and self.decoder_pe_dropout_ratio <= 0.5

        self.decoder_pd_dropout = decoder_pd_dropout
        self.decoder_pd_mixup = decoder_pd_mixup
        self.decoder_pd_dropout_ratio = decoder_pd_dropout_ratio
        assert self.decoder_pd_dropout_ratio >= 0. and self.decoder_pd_dropout_ratio <= 0.5

    def forward(self, x, encoder_out, pe_out=None, pd_out=None, src_padding_mask=None, tgt_padding_mask=None):
        """
        :param mandatory Tensor[batch_size, tgt_seq_len, d_model] x: embedding of previous decoder layer
        :param mandatory Tensor[batch_size, src_seq_len, d_model] encoder_out: embedding of final encoder layer
        :param mandatory Tensor[batch_size, src_seq_len, d_pe] pe_out: embedding of pe
        :param optional Tensor[batch_size, 1, src_seq_len] src_padding_mask: padding mask for key from decoder
        :param optional Tensor[batch_size, 1, tgt_seq_len] tgt_padding_mask: padding mask for key from encoder, pe
        """
        self_attention = self.self_attention(x, x, x, padding_mask=tgt_padding_mask)
        pd_dec_attention = self.pd_dec_attention(
            x, pd_out, pd_out, padding_mask=tgt_padding_mask
        ) if self.pd_dec_attention is not None else None
        pd_ratios = self.get_pd_ratios()
        pdfused_attention_residual = self.pdfused_attention_residual(
            pd_ratios[0]*self_attention + pd_ratios[1]*pd_dec_attention if pd_dec_attention is not None else self_attention, 
            x
        )

        enc_dec_attention = self.encoder_decoder_attention(
            pdfused_attention_residual, encoder_out, encoder_out, padding_mask=src_padding_mask
        )
        pe_dec_attention = self.pe_dec_attention(
            pdfused_attention_residual, pe_out, pe_out, padding_mask=src_padding_mask
        ) if self.pe_dec_attention is not None else None
        pe_ratios = self.get_pe_ratios()
        pefused_attention = self.pefused_attention_residual(
            pe_ratios[0]*enc_dec_attention + pe_ratios[1]*pe_dec_attention if pe_dec_attention is not None else enc_dec_attention,
            pdfused_attention_residual
        )
        return self.feed_forward_residual(
            self.feed_forward(pefused_attention),
            pefused_attention
        )
    
    def get_pe_ratios(self):
        if self.decoder_pe_dropout:
            frand = float(uniform(0, 1))
            if self.decoder_pe_mixup and self.training:
                return [frand, 1 - frand]
            if frand < self.decoder_pe_dropout_ratio and self.training:
                return [1, 0]
            elif frand > 1 - self.decoder_pe_dropout_ratio and self.training:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [0.5, 0.5]

    def get_pd_ratios(self):
        if self.decoder_pd_dropout:
            frand = float(uniform(0, 1))
            if self.decoder_pd_mixup and self.training:
                return [frand, 1 - frand]
            if frand < self.decoder_pd_dropout_ratio and self.training:
                return [1, 0]
            elif frand > 1 - self.decoder_pd_dropout_ratio and self.training:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [0.5, 0.5]


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_Q_in, d_K_in, d_V_in, d_k, d_v, dropout, masking=False):
        """
        :param mandatory int num_heads
        :param mandatory int d_Q_in
        :param mandatory int d_K_in
        :param mandatory int d_V_in
        :param mandatory int d_k
        :param mandatory int d_v
        """
        super(MultiHeadAttention, self).__init__()
        self.masking = masking
        self.heads = nn.ModuleList([
            Attention(d_Q_in=d_Q_in, d_K_in=d_K_in, d_V_in=d_V_in, d_k=d_k, d_v=d_v, dropout=dropout) 
            for _ in range(num_heads)
        ])
        self.linear = Linear(num_heads * d_v, d_Q_in)

    def forward(self, Q, K, V, padding_mask=None):
        """
        :param mandatory Tensor[batch_size, Q_seq_length, d_Q_in] Q: query
        :param mandatory Tensor[batch_size, K_seq_length, d_K_in] K: key
        :param mandatory Tensor[batch_size, K_seq_length, d_V_in] V: value
        :return Tensor[batch_size, Q_seq_length, d_v]
        """
        attention_mask = None
        if self.masking:
            attention_mask = torch.ones(Q.size(0), Q.size(1), K.size(1), device=Q.device).triu(diagonal=1).type(torch.bool) # [batch_size, Q_seq_length, K_seq_length]
        concat_heads = torch.cat([
            h(Q, K, V, attention_mask=attention_mask, padding_mask=padding_mask)[0]
            for h in self.heads
        ], dim=-1) # [batch_size, Q_seq_length, num_heads * d_v]
        return self.linear(concat_heads) # [batch_size, Q_seq_length, d_model]


class Attention(nn.Module):
    def __init__(self, d_Q_in, d_K_in, d_V_in, d_k, d_v, dropout):
        """
        :param mandatory int d_Q_in
        :param mandatory int d_K_in
        :param mandatory int d_V_in
        :param mandatory int d_k
        :param mandatory int d_v
        """
        super(Attention, self).__init__()
        self.q = Linear(d_Q_in, d_k)
        self.k = Linear(d_K_in, d_k)
        self.v = Linear(d_V_in, d_v)
        self.dropout = nn.Dropout(dropout)
        self.linear = Linear(d_v, d_v)

    def forward(self, Q, K, V, attention_mask=None, padding_mask=None, special_token_mask=None):
        """
        :param mandatory Tensor[batch_size, Q_seq_length, d_Q_in] Q: query
        :param mandatory Tensor[batch_size, K_seq_length, d_K_in] K: key
        :param mandatory Tensor[batch_size, K_seq_length, d_V_in] V: value
        :param optional Tensor[batch_size, 1, K_seq_length] padding_mask: masking for padding
        :param optional Tensor[batch_size, Q_seq_length, K_seq_length] attention_mask: masking for attention
        :param optional Tensor[batch_size, 1, K_seq_length] special_token_mask: masking for special tokens (oov_token, number, named entity,...) -> used in PGN
        :return Tensor[batch_size, Q_seq_length, d_v] context_matrix, Tensor[batch_size, Q_seq_length, K_seq_length] attention_distribution
        """
        query = self.q(Q) # [batch_size, Q_seq_length, d_k]
        key = self.k(K) # [batch_size, K_seq_length, d_k]
        value = self.v(V) # [batch_size, K_seq_length, d_v]
        scale = query.size(-1) ** 0.5 # scalar
        attention_weight = query.bmm(key.transpose(1,2)) / scale # [batch_size, Q_seq_length, K_seq_length]
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask, float("-Inf")) # repalce masked position by -inf
        if padding_mask is not None:
            # do not attend to padding symbols
            attention_weight = attention_weight.masked_fill(padding_mask, float("-Inf")) # repalce masked position by -inf
        if special_token_mask is not None:
            attention_weight = attention_weight.masked_fill(~special_token_mask, float("-Inf")) # repalce masked position by -inf
        score = F.softmax(attention_weight, dim=-1) # [batch_size, Q_seq_length, K_seq_length]
        score = score.nan_to_num()
        score = self.dropout(score)
        return self.linear(score.bmm(value)), score # [batch_size, Q_seq_length, d_v], [batch_size, Q_seq_length, K_seq_length]


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, padding_idx):
        """
        :param madatory int vocab_size
        :param madatory int d_model
        """
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.embedding_scale = math.sqrt(d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx

        self._init_weights()


    def forward(self, x):
        """
        :param mandatory Tensor[batch_size, seq_length] x: each sequence is a "list" of vocab index [0, vocab_size)
        :return Tensor[batch_size, seq_length, d_model]
        """
        return self.dropout(self.embedding(x)*self.embedding_scale + self.positional_encoding(x))

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.embedding.embedding_dim ** -0.5)
        nn.init.constant_(self.embedding.weight[self.padding_idx], 0)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        """
        :param madatory int d_model
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
    
    def get_angles(self, positions, indexes):
        d_model_tensor = torch.tensor([[self.d_model]], dtype=torch.float32, device=positions.device)
        angle_rates = torch.pow(10000, (2 * (indexes // 2)) / d_model_tensor)
        return positions / angle_rates

    def forward(self, x):
        """
        :param mandatory Tensor[batch_size, seq_len] x
        :return Tensor[batch_size, seq_len, d_model]
        """
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(1) # [seq_len, 1]
        indexes = torch.arange(self.d_model, device=x.device).unsqueeze(0) # [1, d_model]
        angles = self.get_angles(positions, indexes) # [seq_len, d_model]
        angles[:, 0::2] = torch.sin(angles[:, 0::2]) # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = torch.cos(angles[:, 1::2]) # apply cos to odd indices in the tensor; 2i
        position_encoding = angles.unsqueeze(0).repeat(x.size(0), 1, 1) # [batch_size, seq_len, d_model]
        return position_encoding



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        :param mandatory int d_model
        :param mandatory int d_ff
        """
        super(FeedForward, self).__init__()
        self.first_linear = Linear(d_model, d_ff)
        self.second_linear = Linear(d_ff, d_model)

    def forward(self, x):
        """
        :param mandatory Tensor[batch_size, seq_len] x
        :return Tensor[batch_size, seq_len, d_model]
        """
        x = self.first_linear(x)
        x = F.relu(x)
        return self.second_linear(x)

class Residual(nn.Module):
    def __init__(self, d_model, dropout):
        """
        :param mandatory nn.Module sublayer
        :param mandatory int d_model
        """
        super(Residual, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_non_dropout):
        """
        :param mandatory Tensor[batch_size, seq_len] x
        :return Tensor[batch_size, seq_len, d_model] position_encoding
        """
        return self.layer_norm(self.dropout(x) + x_non_dropout)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        self._init_weights()

    def forward(self, x):
        return self.linear(x)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
