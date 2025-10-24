import torch
from typing import Optional

from ..modules.position_encodings import getPositionEncoder
from ..config import ModelConfig, EncoderConfig, DecoderConfig
from ..constants import FORWARD_MODES
from ..modules.encoder import TextEncoder
from ..modules.decoder import TextDecoder

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

        encoderOut = self.encoder(srcTokens) # [B, S, D]
        decoder: TextDecoder = self.decoder[targetLang] # type: ignore

        if mode == "reconstruct":
            return decoder(srcTokens, encoderOut) # reconstruct source
        elif mode == "translate":
            if tgtTokens is not None:
                return decoder(tgtTokens, encoderOut) # generate target
            else:
                return self.generate(decoder, encoderOut)
        else:
            raise ValueError(f"Unsupported mode: {mode}, supported modes are: {FORWARD_MODES.__args__}") # Print supported modes
    
    def generate(
        self,
        decoder: TextDecoder,
        encoderOut: torch.Tensor,
    ) -> torch.Tensor:
        B = encoderOut.size(0)
        device = encoderOut.device
        generated = torch.full(
            (B, 1),
            self.config.startToken,
            dtype = torch.long,
            device = device,
        )

        for _ in range(self.config.maxSeqLen):
            logits = decoder(generated, encoderOut) # [B, T, V]
            nextToken = logits[:, -1].argmax(dim = -1, keepdim = True) # [B, 1]
            generated = torch.cat([generated, nextToken], dim = 1)

            if hasattr(self.config, 'eosToken') and (nextToken == self.config.eosToken).all():
                break
        
        return generated
