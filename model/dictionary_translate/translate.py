import os
import re
import json
import string
from tqdm import tqdm

from config.config import Configuration
from services.vn_core_service import VnCoreService


class Translator:
    def __init__(self, config: Configuration, vn_core_service: VnCoreService):
        self.config = config
        self.punctuation = string.punctuation + "-"
        self.dictionary = self.load_dictionary(config.vi_ba_dictionary_path, config.vi_ba_train_data_folder)
        self.load_syn_data()
        # self.annotator = VnCoreNLP(address=vn_core_host, port=vn_core_port)
        vi_words = list(self.dictionary.keys())
        vi_words.sort(key=lambda w: len(w), reverse=True)
        self.re = re.compile(" | ".join(vi_words))
        self.vn_core_service = vn_core_service

    def load_syn_data(self):
        def get_all_syn(word, w_set=None):
            _syn_words = [item for syns in data[word]["syn"].values() for item in syns]
            return _syn_words
            # if w_set is None:
            #     w_set = set()
            # if word in w_set:
            #     return w_set
            # else:
            #     w_set.add(word)
            #     _syn_words = [item for syns in data[word]["syn"].values() for item in syns]
            #     for syn_word in _syn_words:
            #         get_all_syn(syn_word, w_set)
            #     return w_set

        print("LOAD SYN DATA")
        data = json.load(open(self.config.synonyms_path, "r", encoding="utf8"))
        done = False
        while not done:
            dict_length = len(self.dictionary)
            vi_words = list(self.dictionary.keys())
            for vi_word in vi_words:
                if vi_word in data:
                    syn_words = get_all_syn(vi_word)
                    for w in syn_words:
                        for c in string.punctuation:
                            w = w.replace(c, "")
                        w = w.strip()
                        if w in self.dictionary or len(w) == 0:
                            continue
                        if w == "toán":
                            continue
                        self.dictionary[w] = self.dictionary[vi_word]
                        # up_vi = w[0].upper() + w[1:]
                        # up_ba = self.dictionary[vi_word][0].upper() + self.dictionary[vi_word][1:]
                        # self.dictionary[up_vi] = up_ba
            done = len(self.dictionary) == dict_length

    def load_dictionary(self, dictionary_path, train_data_folder):
        def get_text(item):
            item = item.replace("\n", "").strip().replace("(", " ").replace(")", " ")
            while "  " in item:
                item = item.replace("  ", " ")
            return item.strip()

        def read_extra_data(data_folder):
            vi_paths = [os.path.join(data_folder, f"{mode}.vi") for mode in ["train", "valid"]]
            ba_paths = [os.path.join(data_folder, f"{mode}.ba") for mode in ["train", "valid"]]
            extra_dict = {}
            for vi_path, ba_path in zip(vi_paths, ba_paths):
                for vi_line, ba_line in zip(open(vi_path, "r", encoding="utf8").readlines(),
                                            open(ba_path, "r", encoding="utf8").readlines()):
                    for c in self.punctuation:
                        vi_line = vi_line.replace(c, f" ")
                        ba_line = ba_line.replace(c, f" ")
                    while "  " in vi_line:
                        vi_line = vi_line.replace("  ", " ")
                    while "  " in ba_line:
                        ba_line = ba_line.replace("  ", " ")
                    vi_line = vi_line.rstrip()
                    ba_line = ba_line.rstrip()
                    if len(vi_line.split()) <= 4 and vi_line not in extra_dict:
                        extra_dict[vi_line] = ba_line
            return extra_dict

        files = os.listdir(dictionary_path)
        vi_files = [os.path.join(dictionary_path, item) for item in files if "vi" in item]
        ba_files = [os.path.join(dictionary_path, item.replace("vi", "bana")) for item in files if "vi" in item]

        vi_data = [get_text(item) for file in vi_files for item in open(file, "r", encoding="utf8").readlines()]
        ba_data = [get_text(item) for file in ba_files for item in open(file, "r", encoding="utf8").readlines()]
        # vi_file = os.path.join(dictionary_path, [item for item in files if "vi" in item][0])
        # ba_file = os.path.join(dictionary_path, [item for item in files if "bana" in item][0])
        # vi_data = [get_text(item) for item in open(vi_file, "r", encoding="utf8").readlines()]
        # ba_data = [get_text(item) for item in open(ba_file, "r", encoding="utf8").readlines()]
        dictionary = {}
        for vi, ba in zip(vi_data, ba_data):
            vi = vi.strip()
            ba = ba.strip()
            vi_words = vi.split(",")
            ba_word = ba.split(",")[0]
            ba_word = ba_word.strip()
            for vi_word in vi_words:
                vi_word = vi_word.strip()
                if len(vi_word) == 0 or vi_word in dictionary:
                    continue
                dictionary[vi_word] = ba_word
            # up_vi = vi[0].upper() + vi[1:]
            # up_ba = ba[0].upper() + ba[1:]
            # dictionary[up_vi] = up_ba
        extra_dictionary = read_extra_data(train_data_folder)
        for vi, ba in extra_dictionary.items():
            vi = vi.strip()
            ba = ba.strip()
            if vi not in dictionary:
                dictionary[vi] = ba
                # up_vi = vi[0].upper() + vi[1:]
                # up_ba = ba[0].upper() + ba[1:]
                # dictionary[up_vi] = up_ba
        return dictionary

    def __call__(self, text):
        sentences = self.annotator.tokenize(text)
        out = []
        for words in sentences:
            sentence_out = []
            for word in words:
                word = word.replace("_", " ")
                sentence_out.append(self.dictionary.get(word.lower(), word))
            sentence = " ".join(sentence_out)
            out.append(sentence)
        return ". ".join(out)

    def translate_word_(self, word):
        output = self.dictionary.get(word.lower())
        if output is None and ("(" in word or ")" in word):
            word = word.replace("(", " ").replace(")", " ")
            while "  " in word:
                word = word.replace("  ", " ")
            output = self.dictionary.get(word.lower())
        if output is None:
            words = word.split()
            if len(words) > 1:
                output = []
                for w in words:
                    translate_word = self.dictionary.get(w.lower())
                    if translate_word is None:
                        output = None
                        break
                    else:
                        output.append(translate_word)
                if output is not None:
                    output = " ".join(output)
        return output

    def _translate_word(self, word, ners, replace_all=True):
        word = f" {word} "
        word = word.replace(" ạ ", " ")
        for c in self.punctuation:
            word = word.replace(c, f" {c} ")

        ner_words = set(item for ner in ners for item in ner.split())
        word_set = set(w for w in word.split())
        for w in word_set:
            if w not in ner_words:
                word = word.replace(w, w.lower())

        word_ = word
        for ner in ners:
            word_ = word_.replace(ner, "")

        n_candidates = 0
        done = False
        while not done:
            words = list(self.re.findall(word))
            print(f"|{words}|>>>|{word}|")
            words = [item for item in words if f" {item.strip()} " in word]

            for w in words:
                word_ = word_.replace(w, " ")
                word = word.replace(w, f"  {self.dictionary.get(w.strip(), w.strip())}  ")

            if len(words) == n_candidates:
                done = True
            n_candidates = len(words)

        for c in self.punctuation:
            word_ = word_.replace(c, "")
        print(f"++++++++++++{word_}")
        if len(word_.strip()) > 0 and replace_all:
            return None
        else:
            while "  " in word:
                word = word.replace("  ", " ")
            word = word.strip()
        return word

    def translate_word(self, word, ners=None, replace_all=True):
        if ners is None:
            ners = self.vn_core_service.get_ner(word)
        output = self._translate_word(word, ners, replace_all)
        if output is None:
            for c in self.punctuation:
                word = word.replace(c, " ")
            while "  " in word:
                word = word.replace("  ", " ")
            word = word.strip()
            output = self._translate_word(word, ners)

        return output

    # print(words)


def main():
    config = Configuration()
    vn_core_nlp = VnCoreService(config)
    translator = Translator(config, vn_core_service=vn_core_nlp)
    # data_path = "transformer-bert-pgn/data/vi-ba/test.vi"
    src_path = "translated_data/loanformer.txt"
    src_data = open(src_path, "r", encoding="utf8").readlines()
    src_data = [": ".join(item.rstrip().split(": ")[1:]) for item in src_data if item[:4] == "--|S"]
    c = 0
    for i, line in enumerate(tqdm(src_data)):
        line = line.rstrip()
        # ners = vn_core_nlp.get_ner(line)
        ners = []
        output = translator.translate_word(line, ners, replace_all=False)
        with open("translated_data/dict_translate.txt", "a", encoding="utf8") as f:
            f.write(f"--|P-{i}: {output}\n")
        if output is None:
            c += 1
            print(f"|{line}|")
    print(translator.translate_word("những con trâu (ấy)"))
    print(c)


if __name__ == "__main__":
    main()
