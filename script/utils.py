
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