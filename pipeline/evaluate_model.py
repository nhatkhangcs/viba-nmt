# if model == ModelTypes.TRANSFORMER:
#     return self.transformer_model([text])[0]
# elif model == ModelTypes.PHOBERT_FUSED:
#     return self.pho_bert_fused_model([text])[0]
# elif model == ModelTypes.LOAN_FORMER:
#     return self.loan_former_model([text])[0]
# elif model == ModelTypes.BARTPHO_ENCODER_PGN:
#     return self.bartpho_encoder_pgn_model([text])[0]
# elif model == ModelTypes.PE_PD_PGN:
#     return self.pe_pd_pgn_model([text])[0]
# elif model == ModelTypes.BART_PHO:
#     text = self.preprocess(text)
#     text = self.add_dot(text)
#     return self.bart_pho_model(text)

# read 6 data files from Data folder by function pointer, namely:
# - test.ba and test.vi
# - train.ba and train.vi
# - valid.ba and valid.vi

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

from transformers import AutoTokenizer, AutoModel

from sacrebleu import corpus_bleu
from common.model_types import ModelTypes
from config.config import Configuration
from services.vn_core_service import VnCoreService
from pipeline.transformer_pgn_translate import TransformerPGNTranslator

class read_data:
    def read_train_data(read_file):
        with open(os.path.join('Data', read_file), encoding='utf-8') as f:
            return f.read().split('\n')
        
    def read_test_data(read_file):
        with open(os.path.join('Data', read_file), encoding='utf-8') as f:
            return f.read().split('\n')
        
    def read_valid_data(read_file):
        with open(os.path.join('Data', read_file), encoding='utf-8') as f:
            return f.read().split('\n')
        
class eval:
    def __init__(self, config: Configuration):
        self.config = config
        self.vn_core_service = VnCoreService(config)
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        pho_bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.loan_former_model = TransformerPGNTranslator(config, tokenizer, self.vn_core_service, pho_bert,
                                                           ModelTypes.LOAN_FORMER)
        self.pho_bert_fused_model = TransformerPGNTranslator(config, tokenizer, self.vn_core_service, pho_bert,
                                                              ModelTypes.PHOBERT_FUSED)
        self.transformer_model = TransformerPGNTranslator(config, tokenizer, self.vn_core_service, pho_bert,
                                                           ModelTypes.TRANSFORMER)
        self.ModelType = None

    def translate_sent(self, text, model=ModelTypes.COMBINED):
        # text: a sentence in Vietnamese
        # model: a model type

        if model == ModelTypes.TRANSFORMER:
            return self.transformer_model([text])[0]
        elif model == ModelTypes.PHOBERT_FUSED:
            return self.pho_bert_fused_model([text])[0]
        elif model == ModelTypes.LOAN_FORMER:
            return self.loan_former_model([text])[0]
        else:
            return None
    
    # def process(self, text, model):
    #     vi_paragraphs = text.split('\n')
    #     ba_paragraphs = []
    #     for paragraph in vi_paragraphs:
    #         sents = self.vn_core_service.tokenize(paragraph)
    #         sents = [" ".join(sent).replace("_", " ") for sent in sents]
    #         translated_sentences = [self.translate_sent(sent, model) for sent in sents]
    #         translated_paragraph = " ".join(translated_sentences)
    #         ba_paragraphs.append(translated_paragraph if paragraph != '' else '')
    #     return ba_paragraphs, ""
    
    def evaluate(self, ba, vi):
        vi_paragraphs = vi
        ba_paragraphs = []
        for vi_paragraphs in vi_paragraphs:
            print("Paragraph:", vi_paragraphs)
            sents = self.vn_core_service.tokenize(vi_paragraphs)
            print("After tokenize:", sents)
            sents = [" ".join(sent).replace("_", " ") for sent in sents]
            print("After join:", sents)
            translated_sentences = [self.translate_sent(sent, self.ModelType) for sent in sents]
            translated_paragraph = " ".join(translated_sentences)
            print("Translated paragraph:", translated_paragraph)
            ba_paragraphs.append(translated_paragraph if vi_paragraphs != '' else '')
        bleu = corpus_bleu(ba_paragraphs, [ba], lowercase=True)
        return bleu.score
    
if __name__ == '__main__':
    train_ba = read_data.read_train_data('train.ba')
    train_vi = read_data.read_train_data('train.vi')
    test_ba = read_data.read_test_data('test.ba')
    test_vi = read_data.read_test_data('test.vi')
    valid_ba = read_data.read_valid_data('valid.ba')
    valid_vi = read_data.read_valid_data('valid.vi')

    # evaluate model
    print("Evaluate model")
    eval_model = eval(Configuration)
    eval_model.ModelType = ModelTypes.TRANSFORMER
    print(eval_model.evaluate(test_ba, test_vi))


    
    # use sacreBLEU to evaluate the translation quality
    
    
