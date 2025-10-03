from src.config import DecoderConfig, EncoderConfig, ModelConfig
from src.model import TextTransformerModel
import torch
from torchview import draw_graph

from src.utils import makeTrainParser, TrainArgs, getTokenizer
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

    tokenizer = getTokenizer(args.tokenizer)
    
    config.vocabSize = tokenizer.vocab_size

    if args.train_phase == "autoencoder":
        from src.train_auto_encoder import startTrain

        startTrain(
            root = args.data_root,
            languages = languages,
            tokenizer = tokenizer,
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
            root = args.data_root,
            languages = languages,
            tokenizer = tokenizer,
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