from src.config import DecoderConfig, EncoderConfig, ModelConfig
from src.model import TextTransformerModel
import torch
from torchview import draw_graph

from src.utils import makeTrainParser, TrainArgs
from src.config import getModelConfig

def main():
    parser = makeTrainParser()
    args: TrainArgs = parser.parse_args() # type: ignore
    
    languages = (args.src_language, args.tgt_language)
    assert languages[0] != languages[1], f"Src lang ({languages[0]}) is the same as tgt lang ({languages[1]})"
    config = getModelConfig(
        args.model_config,
        languages = languages,
        vocabSize = 5000
    )

    if args.train_phase == "autoencoder":
        from src.train_auto_encoder import startTrain

        startTrain(
            languages = languages,
            tokenizer = torch.nn.Identity(), # Pass tokenizer in here
            modelConfig = config,
            numEpochs = args.num_epochs,
            checkpointDir = args.checkpoint_path,
            shouldLog = args.log,
            batchSize = args.batch_size,
            saveInterval = args.save_interval
        )
        
    else:
        from src.train_back_translation import startTrain
        startTrain(
            languages = languages,
            tokenizer = torch.nn.Identity(),
            modelConfig = config,
            numEpochs = args.num_epochs,
            checkpointDir = args.checkpoint_path,
            shouldLog = args.log,
            batchSize = args.batch_size,
            saveInterval = args.save_interval,
            autoencoderCheckpoint = args.autoencoder_checkpoint
        )


if __name__ == "__main__":
    main()