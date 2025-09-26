import torch
from typing import Optional

from .position_encodings import getPositionEncoder
from .config import ModelConfig, EncoderConfig, DecoderConfig
from .constants import FORWARD_MODES
from .encoder import TextEncoder
from .decoder import TextDecoder

class TextTransformerModel(torch.nn.Module):
    def __init__(
        self,
        modelConfig: ModelConfig = ModelConfig(),
    ):
        super().__init__()
        self.config = modelConfig
        
        sharedPosEncoder = getPositionEncoder(
            self.config.peType, 
            embedDim = self.config.embedDim,
            maxSeqLen = self.config.maxSeqLen,
            nHeads = self.config.nHeads,
        )
        self.encoder = TextEncoder(
            modelConfig, 
            sharedPosEncoder = sharedPosEncoder,
            vocabSize = modelConfig.vocabSize
        )
        sharedEmbedding = self.encoder.tokenEmbedding
        self.decoder = torch.nn.ModuleDict({
            lang: TextDecoder(
                modelConfig, 
                sharedEmbedding = sharedEmbedding,
                sharedPosEncoder = sharedPosEncoder,
                vocabSize = modelConfig.vocabSize
            ) for lang in modelConfig.languages 
        })

    def forward(
        self, 
        srcTokens: torch.Tensor,
        tgtTokens: Optional[torch.Tensor],
        targetLang: str,
        mode: FORWARD_MODES,
    ) -> torch.Tensor:
        # srcTokens: [B, S]
        # tgtTokens: [B, T]
        # mode: "translate" for back-translation and "reconstruct" for denoising and reconstruction
        assert (tgtTokens is None) == (mode == "reconstruct"), f"tgtTokens is None: {tgtTokens is None} and mode is {mode}"

        encoderOut = self.encoder(srcTokens) # [B, S, D]

        if mode == "reconstruct":
            decoder = self.decoder[targetLang]
            return decoder(srcTokens, encoderOut) # reconstruct source
        elif mode == "translate":
            decoder = self.decoder[targetLang]
            return decoder(tgtTokens, encoderOut) # generate target
        else:
            raise ValueError(f"Unsupported mode: {mode}, supported modes are: {FORWARD_MODES.__args__}") # Print supported modes