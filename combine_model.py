import time
import os
from tqdm import tqdm
import requests
from dictionary_translate.translate import Translator


class CombineModel:
    def __init__(self):
        self.session = requests.session()
        self.base_url = "https://vi-ba-nmt-api-proxy.herokuapp.com/translate"
        self.translator = Translator("dictionary_translate/dictionary", "dictionary_translate/data")

    def make_request(self, text, model_name):
        done = False
        output = ""
        c = 1
        while not done and c > 0:
            try:
                output = self.session.post(self.base_url,
                                           json={"text": text, "model": model_name},
                                           headers={"Content-Type": "application/json"}).json()["ResultObj"]["tgt"][0]
                done = True
            except Exception as e:
                self.session = requests.session()
                print("ERROR", e, text)
                time.sleep(1)
                c -= 1
        return output

    def __call__(self, text, model_name="Loanformer"):
        if model_name in ["Loanformer", "PhoBERT-fused NMT", "Transformer"]:
            # output = translate(text, "Loanformer")
            output = self.make_request(text, model_name)
            return output
        else:
            return self.translator.translate_word(text)


def load_old(data_path):
    data_dict = {}
    for line in open(data_path, "r", encoding="utf8").readlines()[1:]:
        line = line.rstrip()
        len, vi, dict_trans, loan, pho_fuse, transformer = line.split("||")

        data_dict[vi] = {
            "len": len.strip(),
            "vi": vi.strip(),
            "Dictionary": dict_trans.strip(),
            "Loanformer": loan.strip(),
            "PhoBERT-fused NMT": pho_fuse.strip(),
            "Transformer": transformer.strip()
        }
    return data_dict


if __name__ == "__main__":
    model = CombineModel()
    saved_data = load_old("dictionary_translate/data/translated.csv")

    test_vi = [item.rstrip() for item in open("dictionary_translate/data/test.vi", "r", encoding="utf8").readlines()]
    if not os.path.exists("dictionary_translate/data/translated_1.csv"):
        with open("dictionary_translate/data/translated_1.csv", "w", encoding="utf8") as f:
            f.write("len||vi||Dictionary||Loanformer||PhoBERT-fused NMT||Transformer\n")
    for vi_text in tqdm(test_vi):
        vi_text = vi_text.strip()
        if vi_text in saved_data:
            output = [str(saved_data[vi_text]["len"]), vi_text]
            for m_name in ["Dictionary", "Loanformer", "PhoBERT-fused NMT", "Transformer"]:
                if len(saved_data[vi_text][m_name]) == 0:
                    saved_data[vi_text][m_name] = model(vi_text, m_name)
                output.append(saved_data[vi_text][m_name])
        else:
            output = [str(len(vi_text.split())), vi_text]
            for m_name in ["Dictionary", "Loanformer", "PhoBERT-fused NMT", "Transformer"]:
                output.append(model(vi_text, m_name))
        with open("dictionary_translate/data/translated_1.csv", "a", encoding="utf8") as f:
            f.write("||".join(output) + "\n")
