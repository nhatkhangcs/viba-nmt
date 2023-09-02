import logging
from time import sleep

from vncorenlp import VnCoreNLP

from config.config import Configuration


class VnCoreService:
    def __init__(self, config: Configuration):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.vn_core_nlp = self.init_vn_core()

    def init_vn_core(self):
        return VnCoreNLP(address=self.config.vn_core_nlp_address, port=self.config.vn_core_nlp_port)

    def ner(self, text):
        result = None
        while result is None:
            try:
                result = self.vn_core_nlp.ner(text)
            except Exception as e:
                self.logger.error(f"Cannot make NER request to NLP Core. Error: {e}")
                self.init_vn_core()
                sleep(1)
        return result

    def tokenize(self, text):
        result = None
        while result is None:
            try:
                result = self.vn_core_nlp.tokenize(text)
            except Exception as e:
                self.logger.error(f"Cannot make TOKENIZE request to NLP Core. Error: {e}")
                self.init_vn_core()
                sleep(1)
        return result

    def has_ner(self, text):
        output = self.ner(text)
        for sent in output:
            for _, label in sent:
                if "PER" in label:
                    return True
        return False

    def get_ner(self, text):
        ners = []
        output = self.ner(text)
        for sent in output:
            for text, label in sent:
                if "B-" in label:
                    ners.append(text)
                elif "I-" in label:
                    ners[-1] += " " + text
        ners = list(set(ners))
        for i, ner in enumerate(ners):
            ner = ner.replace("_", " ")
            while "  " in ner:
                ner = ner.replace("  ", " ")
            ners[i] = ner
        return ners
