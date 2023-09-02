import logging

import torch
from transformers import AutoTokenizer

from config.config import Configuration
from model.vi_ba_bartpho.custom_mbart_model import CustomMbartModel


class BartPhoTranslator:
    def __init__(self, config: Configuration):
        self.logger = logging.getLogger(__class__.__name__)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.vi_ba_bart_pho_checkpoint)
        self.model = CustomMbartModel.from_pretrained(self.config.vi_ba_bart_pho_checkpoint)

    def __call__(self, text):
        input_ids = self.tokenizer(text).input_ids
        input_ids = torch.tensor([input_ids])
        outputs = self.model.generate(input_ids=input_ids, num_beams=5, max_length=1024, num_return_sequences=1)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output
