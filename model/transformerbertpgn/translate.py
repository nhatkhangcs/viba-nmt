import os

from vncorenlp import VnCoreNLP
from transformers import AutoModel, AutoTokenizer

from config.config import Configuration
from model.transformerbertpgn.Dictionary import Dictionary
from model.transformerbertpgn.model import NMT
from model.transformerbertpgn.model_pepdfused import NMT as PEPDfusedNMT
from model.transformerbertpgn.model_bartphoencoderfused import NMT as BartphoEncoderfusedNMT
from model.transformerbertpgn.Loss import Loss
from model.transformerbertpgn.utils import *


print("BASE MODEL INITIALIZING")

# annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
# phobert = AutoModel.from_pretrained("vinai/phobert-base")

print("BASE MODEL INITIALIZED")

# get device
device = 'cpu'


def get_model(dictionary_path, checkpoint_path, tokenizer, annotator, bert=None,
              d_model=512, d_ff=2048, num_heads=8, num_layers=6, dropout=0.1, d_bert=None,
              use_pgn=False, use_ner=False, max_src_len=256, max_tgt_len=256, device='cpu'):
    # load dictionary
    dictionary = Dictionary(tokenizer=tokenizer)
    dictionary.add_from_file(dictionary_path)
    dictionary.build_dictionary()
    print(f'--|Vocab size: {len(dictionary)}')

    # init criterion
    criterion = Loss(ignore_idx=dictionary.token_to_index(dictionary.pad_token), smoothing=0.1)

    # load model
    print(checkpoint_path)
    print(os.path.exists(checkpoint_path))
    model = NMT.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        dictionary=dictionary,
        tokenizer=tokenizer,
        annotator=annotator,
        criterion=criterion,
        map_location=torch.device('cpu'),
        strict=False,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        bert=bert,
        d_bert=d_bert,
        use_pgn=use_pgn,
        use_ner=use_ner,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )
    model.eval()
    model.to(device)

    return model

def get_model_bartpho_encoder_fused(dictionary_path, checkpoint_path, tokenizer, annotator, bert=None,
              d_model=512, d_ff=2048, num_heads=8, num_layers=6, dropout=0.1, d_bert=None,
              use_pgn=False, use_ner=False, max_src_len=256, max_tgt_len=256, device='cpu'):
    # load dictionary
    dictionary = Dictionary(tokenizer=tokenizer)
    dictionary.add_from_file(dictionary_path)
    dictionary.build_dictionary()
    print(f'--|Vocab size: {len(dictionary)}')

    # init criterion
    criterion = Loss(ignore_idx=dictionary.token_to_index(dictionary.pad_token), smoothing=0.1)

    # load model
    print(checkpoint_path)
    print(os.path.exists(checkpoint_path))
    model = BartphoEncoderfusedNMT.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        dictionary=dictionary,
        tokenizer=tokenizer,
        annotator=annotator,
        criterion=criterion,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        bert=bert,
        d_bert=d_bert,
        use_pgn=use_pgn,
        use_ner=use_ner,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )
    model.eval()
    model.to(device)

    return model

def get_model_pepdfused(dictionary_path, checkpoint_path, tokenizer, annotator, bart=None,
              d_model=512, d_ff=2048, num_heads=8, num_layers=6, dropout=0.1, d_bart=None,
              use_pgn=False, use_ner=False, max_src_len=256, max_tgt_len=256, device='cpu'):
    # load dictionary
    dictionary = Dictionary(tokenizer=tokenizer)
    dictionary.add_from_file(dictionary_path)
    dictionary.build_dictionary()
    print(f'--|Vocab size: {len(dictionary)}')

    # init criterion
    criterion = Loss(ignore_idx=dictionary.token_to_index(dictionary.pad_token), smoothing=0.1)

    # load model
    print(checkpoint_path)
    print(os.path.exists(checkpoint_path))
    model = PEPDfusedNMT.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        dictionary=dictionary, 
        tokenizer=tokenizer, 
        annotator=annotator, 
        criterion=criterion,
        d_model=d_model, 
        d_ff=d_ff,
        num_heads=num_heads, 
        num_layers=num_layers, 
        dropout=dropout,
        bart=bart,
        d_bart=d_bart,
        use_pgn=use_pgn,
        use_ner=use_ner,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )
    model.eval()
    model.to(device)

    return model


def get_transformer_model(config: Configuration, tokenizer, annotator):
    # get models
    transformer_model = get_model(
        # dictionary_path="./transformerbertpgn/data/vi-ba/dict-synonymaugment.txt",
        # checkpoint_path="./transformerbertpgn/checkpoints/transformer/epoch=17-val_loss=1.40.ckpt",
        dictionary_path=config.transformers_dictionary_path,
        checkpoint_path=config.transformers_checkpoint_path,
        tokenizer=tokenizer,
        annotator=annotator,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        bert=None,
        d_bert=None,
        use_pgn=False,
        use_ner=False,
        max_src_len=256,
        max_tgt_len=256,
        device=device
    )
    return transformer_model


def get_phobert_fused_model(config: Configuration, tokenizer, annotator, phobert):

    phobert_fused_model = get_model(
        # dictionary_path="transformerbertpgn/data/vi-ba/dict-synonymaugment.txt",
        # checkpoint_path="transformerbertpgn/checkpoints/phobert-fused/epoch=20-val_loss=1.24.ckpt",
        dictionary_path=config.pho_bert_fused_dictionary_path,
        checkpoint_path=config.pho_bert_fused_checkpoint_path,
        tokenizer=tokenizer,
        annotator=annotator,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        bert=phobert,
        d_bert=768,
        use_pgn=False,
        use_ner=False,
        max_src_len=256,
        max_tgt_len=256,
        device=device
    )
    return phobert_fused_model


def get_loanformer_model(config: Configuration, tokenizer, annotator, phobert):
    loan_former_model = get_model(
        # dictionary_path="./transformerbertpgn/data/vi-ba/dict-synonymaugment-accent.txt",
        # checkpoint_path="./transformerbertpgn/checkpoints/tbmp-number-mask-accent/epoch=20-val_loss=1.18.ckpt",
        dictionary_path=config.loan_former_dictionary_path,
        checkpoint_path=config.loan_former_checkpoint_path,
        tokenizer=tokenizer,
        annotator=annotator,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        bert=phobert,
        d_bert=768,
        use_pgn=True,
        use_ner=True,
        max_src_len=256,
        max_tgt_len=256,
        device=device
    )
    return loan_former_model

def get_bartpho_encoder_pgn_model(config: Configuration, tokenizer, annotator, phobert):
    bartpho_encoder_pgn_model = get_model_bartpho_encoder_fused(
        dictionary_path=config.bartphoencoder_pgn_dictionary_path,
        checkpoint_path=config.bartphoencoder_pgn_checkpoint_path,
        tokenizer=tokenizer,
        annotator=annotator,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        bert=phobert,
        d_bert=1024,
        use_pgn=True,
        use_ner=True,
        max_src_len=256,
        max_tgt_len=256,
        device=device
    )
    return bartpho_encoder_pgn_model

def get_pe_pd_pgn_model(config: Configuration, tokenizer, annotator, bartpho):
    pe_pd_pgn_model = get_model_pepdfused(
        dictionary_path=config.pe_pd_pgn_dictionary_path,
        checkpoint_path=config.pe_pd_pgn_checkpoint_path,
        tokenizer=tokenizer,
        annotator=annotator,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        bart=bartpho,
        d_bart=1024,
        use_pgn=True,
        use_ner=True,
        max_src_len=256,
        max_tgt_len=256,
        device=device
    )
    return pe_pd_pgn_model


def process_raw_text(text, dictionary, tokenizer, annotator, 
                max_src_len=256, use_pgn=False, use_ner=False, device='cpu'):
    """
    input: list [batch_size] of tensors in different lengths
    output: tensor [batch_size, seq_length]
    """
    src_batch = text
    dictionary_ext = None
    if use_pgn:
        dictionary_ext = Dictionary(tokenizer=tokenizer)
        dictionary_ext.token2index = {**dictionary.token2index}
        dictionary_ext.index2token = {**dictionary.index2token}
        dictionary_ext.vocab_size = dictionary.vocab_size
    src = []
    src_bert = []
    src_ext = [] if use_pgn else None
    src_ne = [] if use_ner else None
    for idx in range(len(src_batch)):
        src_preprocessed = preprocess(annotator, src_batch[idx].strip(), ner=use_ner)
        src_str = " ".join(src_preprocessed['words'])
        src_encode = dictionary.encode(src_str, append_bos=False, append_eos=True)
        src.append(torch.tensor(src_encode['ids']))
        src_bert.append(torch.tensor(tokenizer.encode(src_str)[1:]))
        if use_pgn:
            src_ext.append(torch.tensor(
                dictionary_ext.encode(src_str, append_bos=False, append_eos=True, update=True)['ids']
            ))
        if use_ner:
            src_ne.append(
                torch.tensor(ner_for_bpe(
                    bpe_tokens=src_encode['bpe_tokens'], ne_tokens=src_preprocessed['name_entities'], 
                    get_mask=True, special_tokens=[dictionary.bos_token, dictionary.eos_token]
                ))
            )

    src = pad_sequence(src, padding_value=dictionary.token_to_index(dictionary.pad_token), batch_first=True)
    src_bert = pad_sequence(src_bert, padding_value=tokenizer.pad_token_id, batch_first=True)
    if use_pgn:
        src_ext = pad_sequence(src_ext, padding_value=dictionary_ext.token_to_index(dictionary_ext.pad_token), batch_first=True)
    if use_ner:
        src_ne = pad_sequence(src_ne, padding_value=0, batch_first=True)
    assert src.size(1) == src_bert.size(1)
    # Truncate if seq_len exceed max_src_length
    if src.size(1) > max_src_len:
        src = src[:,:max_src_len]
        src_bert = src_bert[:,:max_src_len]
        if use_pgn:
            src_ext = src_ext[:,:max_src_len]
        if use_ner:
            src_ne = src_ne[:,:max_src_len]
    return {
        'src_raw': src_batch,
        'src': src.to(device), 
        'src_bert': src_bert.to(device), 
        'src_ext': src_ext.to(device) if use_pgn else None,
        'src_ne': src_ne.to(device) if use_ner else None, 
        'dictionary_ext': dictionary_ext,
        'max_oov_len': len(dictionary_ext) - len(dictionary) if use_pgn else None
    }


def translate(vi, selected_model):
    model = None
    if selected_model == 'Transformer':
        model = transformer_model
    elif selected_model == 'PhoBERT-fused NMT':
        model = phobert_fused_model
    elif selected_model == 'Loanformer':
        model = loanformer_model
    else:
        raise Exception('Unsuported model')
        
    input = process_raw_text(
        vi, model.dictionary, model.tokenizer, model.annotator, 
        max_src_len=model.max_src_len, use_pgn=model.model.use_pgn, 
        use_ner=model.model.use_ner, device=model.device
    )

    preds = model.model.inference(
        input['src'], input['src_bert'], input['src_ext'], input['src_ne'], input['max_oov_len'], 
        model.max_tgt_len, model.dictionary.token_to_index(model.dictionary.eos_token)
    )

    # # decode
    # preds = preds.tolist()[0]
    # decode_dict = input['dictionary_ext'] if model.model.use_pgn else model.dictionary
    # tokens = [decode_dict.index_to_token(i) for i in preds]
    # seq = model.tokenizer.convert_tokens_to_string(tokens)
    # ba = model._postprocess(seq)

    # decode
    preds = preds.tolist()
    sequences = []
    decode_dict = input['dictionary_ext'] if model.model.use_pgn else model.dictionary
    for seq_ids in preds:
        tokens = [decode_dict.index_to_token(i) for i in seq_ids]
        seq = model.tokenizer.convert_tokens_to_string(tokens)
        sequences.append(model._postprocess(seq))

    return sequences