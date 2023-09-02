from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import numpy as np
import pandas as pd


truth_path = "translated_data/loanformer.txt"
data_path = "translated_data/loanformer.txt"

loan_former_path = "translated_data/loanformer.txt"
pho_bert_fused_path = "translated_data/phobert_fused.txt"
transformer_path = "translated_data/transformer.txt"
dictionary_path = "translated_data/dict_translate.txt"

original_data = open(data_path, "r", encoding="utf8").readlines()
original_data = [": ".join(item.rstrip().split(": ")[1:]) for item in original_data if item[:4] == "--|S"]
truth_data = open(truth_path, "r", encoding="utf8").readlines()
truth_data = [[": ".join(item.rstrip().split(": ")[1:]).split()] for item in truth_data if item[:4] == "--|T"]

loan_former_data = open(loan_former_path, "r", encoding="utf8").readlines()
loan_former_data = [": ".join(item.rstrip().split(": ")[1:]).split() for item in loan_former_data if item[:4] == "--|P"]

pho_bert_fused_data = open(pho_bert_fused_path, "r", encoding="utf8").readlines()
pho_bert_fused_data = [": ".join(item.rstrip().split(": ")[1:]).split() for item in pho_bert_fused_data if item[:4] == "--|P"]

transformer_data = open(transformer_path, "r", encoding="utf8").readlines()
transformer_data = [": ".join(item.rstrip().split(": ")[1:]).split() for item in transformer_data if item[:4] == "--|P"]

dictionary_data = open(dictionary_path, "r", encoding="utf8").readlines()
dictionary_data = [": ".join(item.rstrip().split(": ")[1:]).split() for item in dictionary_data if item[:4] == "--|P"]


loan_former_scores = [sentence_bleu(references=truth_item, hypothesis=loan_former_item)
                      for truth_item, loan_former_item in zip(truth_data, loan_former_data)]

pho_bert_fused_scores = [sentence_bleu(references=truth_item, hypothesis=pho_bert_fused_item)
                         for truth_item, pho_bert_fused_item in zip(truth_data, pho_bert_fused_data)]

transformer_scores = [sentence_bleu(references=truth_item, hypothesis=transformer_item)
                      for truth_item, transformer_item in zip(truth_data, transformer_data)]

dictionary_scores = [sentence_bleu(references=truth_item, hypothesis=dictionary_item)
                     for truth_item, dictionary_item in zip(truth_data, dictionary_data)]

# print(corpus_bleu(list_of_references=truth_data, hypotheses=loan_former_data))
# print(corpus_bleu(list_of_references=truth_data, hypotheses=pho_bert_fused_data))
# print(corpus_bleu(list_of_references=truth_data, hypotheses=transformer_data))
# print(corpus_bleu(list_of_references=truth_data, hypotheses=dictionary_data))

# print(np.mean(loan_former_scores))
# print(np.mean(dictionary_scores))

all_data = list(zip(original_data, dictionary_scores, transformer_scores, pho_bert_fused_scores, loan_former_scores))
all_data = pd.DataFrame(all_data, columns=["original_data", "dict", "transformers", "phobert_fused", "loan_former"])
all_data.to_csv("translated_data/all.csv")
