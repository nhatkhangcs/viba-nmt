class ModelTypes:
    TRANSFORMER = 'Transformer'
    PHOBERT_FUSED = 'PhoBERT-fused NMT'
    LOAN_FORMER = 'Loanformer'
    BART_PHO = "BartPho"
    COMBINED = "Combined"
    BARTPHO_ENCODER_PGN = 'BARTphoEncoderPGN'
    PE_PD_PGN = 'PE-PD-PGN'

    @classmethod
    def get_models(cls):
        return [
            # ModelTypes.COMBINED,
            # ModelTypes.LOAN_FORMER,
            # ModelTypes.PHOBERT_FUSED,
            # ModelTypes.TRANSFORMER,
            # ModelTypes.BART_PHO,
            ModelTypes.BARTPHO_ENCODER_PGN,
            ModelTypes.PE_PD_PGN
        ]
