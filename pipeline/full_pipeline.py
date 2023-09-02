import string
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])

from transformers import AutoTokenizer, AutoModel

from common.model_types import ModelTypes
from config.config import Configuration
from services.vn_core_service import VnCoreService
from pipeline.transformer_pgn_translate import TransformerPGNTranslator
# from pipeline.pe_pd_fused_translate import PEPDfusedTranslator


class TranslationPipeline:
    def __init__(self, config: Configuration):
        self.config = config
        self.vn_core_service = VnCoreService(config)
        # self.dictionary_translator = Translator(config, self.vn_core_service)
        # self.bart_pho_model = BartPhoTranslator(config)
            # tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
            # pho_bert = AutoModel.from_pretrained("vinai/phobert-base")
            # self.loan_former_model = TransformerPGNTranslator(config, tokenizer, self.vn_core_service, pho_bert,
            #                                                 ModelTypes.LOAN_FORMER)
            # self.pho_bert_fused_model = TransformerPGNTranslator(config, tokenizer, self.vn_core_service, pho_bert,
            #                                                     ModelTypes.PHOBERT_FUSED)
            # self.transformer_model = TransformerPGNTranslator(config, tokenizer, self.vn_core_service, pho_bert,
            #                                                 ModelTypes.TRANSFORMER)
        
        # self.dictionary_translator = Translator(config, self.vn_core_service)
        word_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word", use_fast=False)
        bart_pho = AutoModel.from_pretrained("vinai/bartpho-word")
        self.bartpho_encoder_pgn_model = TransformerPGNTranslator(config, word_tokenizer, self.vn_core_service, bart_pho, ModelTypes.BARTPHO_ENCODER_PGN)
        # self.pe_pd_pgn_model = PEPDfusedTranslator(config, word_tokenizer, self.vn_core_service, bart_pho, ModelTypes.PE_PD_PGN)

    @staticmethod
    def preprocess(text):
        for c in string.punctuation:
            text = text.replace(c, f" {c} ")
        return text

    @staticmethod
    def add_dot(text):
        text = text.strip()
        if text[-1] not in ".:?!":
            text += " ."
        return text

    @staticmethod
    def drop_punctuation(text):
        for c in string.punctuation:
            text = text.replace(c, "")
        text = text.strip()
        return text

    def count_words(self, text, ners):
        for ner in ners:
            text = text.replace(ner, "")
        text = self.drop_punctuation(text)
        return len(text.split())

    def translate_sent(self, text, model=ModelTypes.COMBINED):
        # if model == ModelTypes.TRANSFORMER:
        #     text = self.preprocess(text)
        #     translated = None
        #     ners = self.vn_core_service.get_ner(text)
        #     num_words = self.count_words(text, ners)
        #     if num_words <= 7:
        #         translated = self.dictionary_translator.translate_word(text, ners)
        #     if translated is None:
        #         # print("Model Translate")
        #         text = self.add_dot(text)
        #         if len(ners) > 0:
        #             print("HAS NER")
        #             print("LoanFormer")
        #             return self.loan_former_model([text])[0]
        #         elif num_words <= 7:
        #             print("TRANSFORMER")
        #             return self.transformer_model([text])[0]
        #         else:
        #             print("BARTPhoModel")
        #             return self.bart_pho_model(text)
        #     else:
        #         print("Dictionary Translate")
        #         return translated
        # elif model == ModelTypes.LOAN_FORMER:
        #     return self.loan_former_model([text])[0]
        # else:
        print(model)
        if model == ModelTypes.TRANSFORMER:
            return self.transformer_model([text])[0]
        elif model == ModelTypes.PHOBERT_FUSED:
            return self.pho_bert_fused_model([text])[0]
        elif model == ModelTypes.LOAN_FORMER:
            return self.loan_former_model([text])[0]    
        elif model == ModelTypes.BARTPHO_ENCODER_PGN:
            return self.bartpho_encoder_pgn_model([text])[0]
        elif model == ModelTypes.PE_PD_PGN:
            return self.pe_pd_pgn_model([text])[0]
        elif model == ModelTypes.BART_PHO:
            text = self.preprocess(text)
            text = self.add_dot(text)
            return self.bart_pho_model(text)
        else:
            #BART PHO + DICTIONARY ONLY
            text = self.preprocess(text)
            translated = None
            ners = self.vn_core_service.get_ner(text)
            num_words = self.count_words(text, ners)
            if num_words <= 7:
                translated = self.dictionary_translator.translate_word(text, ners)
            if translated is None:
                # print("Model Translate")
                text = self.add_dot(text)
                return self.bart_pho_model(text)
            else:
                print("Dictionary Translate")
                return translated

    async def __call__(self, text, model=ModelTypes.COMBINED):
        vi_paragraphs = text.split('\n')
        ba_paragraphs = []
        for paragraph in vi_paragraphs:
            sents = self.vn_core_service.tokenize(paragraph)
            sents = [" ".join(sent).replace("_", " ") for sent in sents]
            translated_sentences = [self.translate_sent(sent, model) for sent in sents]
            translated_paragraph = " ".join(translated_sentences)
            ba_paragraphs.append(translated_paragraph if paragraph != '' else '')
        return ba_paragraphs, ""


if __name__ == "__main__":
    from tqdm import tqdm
    config = Configuration()
    pipeline = TranslationPipeline(config)
    vi_data = [item.rstrip() for item in open("checkpoints/dictionary_translate/data/vi_0504_s.txt", "r", encoding="utf8")]
    ba_data = [item.rstrip() for item in open("checkpoints/dictionary_translate/data/bana_0504_s.txt", "r", encoding="utf8")]
    for vi, ba in tqdm(zip(vi_data, ba_data), total=len(vi_data)):
        translated = pipeline(vi)
        if translated is None:
            print(f"\n>>>{vi}|||{ba}<<<\n")
    # output = pipeline("Có bốn buổi: buổi sáng, buổi trưa, buổi chiều và buổi tối")
    # print(output)


