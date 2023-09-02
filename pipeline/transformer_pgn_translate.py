import logging

from common.model_types import ModelTypes
from config.config import Configuration
from model.transformerbertpgn.translate import (
    get_transformer_model,
    get_loanformer_model,
    get_phobert_fused_model,
    # get_bartpho_encoder_pgn_model,
    process_raw_text
)


class TransformerPGNTranslator:
    def __init__(self, config: Configuration, tokenizer, annotator, pho_bert_model, model_type: ModelTypes):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        if model_type == ModelTypes.TRANSFORMER:
            self.model = get_transformer_model(config, tokenizer, annotator)
        elif model_type == ModelTypes.PHOBERT_FUSED:
            self.model = get_phobert_fused_model(config, tokenizer, annotator, pho_bert_model)
        elif model_type == ModelTypes.LOAN_FORMER:
            self.model = get_loanformer_model(config, tokenizer, annotator, pho_bert_model)
        #else:
            #self.model = get_bartpho_encoder_pgn_model(config, tokenizer, annotator, pho_bert_model)

    def __call__(self, text):
        input_data = process_raw_text(
            text, self.model.dictionary, self.model.tokenizer, self.model.annotator,
            max_src_len=self.model.max_src_len, use_pgn=self.model.model.use_pgn,
            use_ner=self.model.model.use_ner, device=self.model.device
        )

        predictions = self.model.model.inference(
            input_data['src'], input_data['src_bert'], input_data['src_ext'], input_data['src_ne'],
            input_data['max_oov_len'], self.model.max_tgt_len,
            self.model.dictionary.token_to_index(self.model.dictionary.eos_token)
        )

        predictions = predictions.tolist()
        sequences = []
        decode_dict = input_data['dictionary_ext'] if self.model.model.use_pgn else self.model.dictionary
        for seq_ids in predictions:
            tokens = [decode_dict.index_to_token(i) for i in seq_ids]
            seq = self.model.tokenizer.convert_tokens_to_string(tokens)
            sequences.append(self.model._postprocess(seq))
        return sequences
