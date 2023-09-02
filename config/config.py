

class Configuration:
    vi_ba_dictionary_path = "checkpoints/dictionary_translate/dictionary"
    vi_ba_train_data_folder = "checkpoints/dictionary_translate/data"
    synonyms_path = "checkpoints/dictionary_translate/synonyms/syn_data.json"
    loan_former_dictionary_path = "checkpoints/dictionary/dict-synonymaugment-accent.txt"
    loan_former_checkpoint_path = "checkpoints/loan_former/epoch=20-val_loss=1.18.ckpt"
    pho_bert_fused_dictionary_path = "checkpoints/dictionary/dict-synonymaugment.txt"
    pho_bert_fused_checkpoint_path = "checkpoints/phobert_fused/epoch=20-val_loss=1.24.ckpt"
    transformers_dictionary_path = "checkpoints/dictionary/dict-synonymaugment.txt"
    transformers_checkpoint_path = "checkpoints/transformers/epoch=17-val_loss=1.40.ckpt"
    vi_ba_bart_pho_checkpoint = "checkpoints/vi_ba_bartpho"
    pe_pd_pgn_dictionary_path = "checkpoints/dictionary/dict-synonymaugment.txt"
    pe_pd_pgn_checkpoint_path = "checkpoints/pe_pd_pgn/PE-PD-PGN.ckpt"
    bartphoencoder_pgn_dictionary_path = "checkpoints/dictionary/dict-synonymaugment-accent.txt"
    bartphoencoder_pgn_checkpoint_path = "checkpoints/bartphoencoder_pgn/BARTphoEncoderPGN.ckpt"
    vn_core_nlp_address = "http://localhost"
    vn_core_nlp_port = 9000

