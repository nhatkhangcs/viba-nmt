import json
from tqdm import tqdm

from script.utils import load_old
from nltk.translate.bleu_score import sentence_bleu


def preprocess(text):
    for c in ".,'" + '"?!()[]{}':
        text = text.replace(c, f" {c} ")
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.strip()
    return text


def main():
    OLD_DATA = "dictionary_translate/data/translated.csv"
    data = load_old(OLD_DATA)
    test_vi = [item.rstrip() for item in open("dictionary_translate/data/test.vi", "r", encoding="utf8").readlines()]
    test_ba = [item.rstrip() for item in open("dictionary_translate/data/test.ba", "r", encoding="utf8").readlines()]
    test_data = {}
    for text_vi, text_ba in tqdm(zip(test_vi, test_ba)):
        if text_vi not in data:
            continue
        test_data[text_vi] = []
        translated_data = data[text_vi]
        label = preprocess(text_ba).split()
        for m_name in ["Dictionary", "Loanformer", "PhoBERT-fused NMT", "Transformer"]:
            translated = translated_data[m_name]
            translated = preprocess(translated).split()
            test_data[text_vi].append(sentence_bleu([label], translated))

    json.dump(test_data, open("dictionary_translate/data/scores.json", "w", encoding="utf8"), indent=4)

    sum = 0
    count = 0
    for item in test_data.values():
        sum += item[1]
        count += 1


if __name__ == "__main__":
    main()
