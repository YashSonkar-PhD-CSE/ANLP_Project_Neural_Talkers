from typing import Tuple, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import os

from .config import ModelConfig
from .datasets import BaseDataset
from .model import TextTransformerModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def trainBackTranslationStage(
    model: TextTransformerModel,
    trainDataset: BaseDataset,
    validDataset: BaseDataset,
    optimizer: torch.optim.Optimizer,
    writer: Optional[SummaryWriter],
    batchSize: int,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    clipNorm: float,
    saveInterval: int = 1,
    checkpointDir: str = "./checkpoints",
    criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    numEpochs: int = 10,
    padToken: int = 0,
    device: torch.device = torch.device("cpu"),
    tokenizer = None
):
    model.to(device)
    model.train()

    globalStep = 0
    srcLang, tgtLang = trainDataset.languages

    for epoch in range(numEpochs):
        logger.info(f"Epoch {epoch + 1}")
        
        for direction in [(srcLang, tgtLang), (tgtLang, srcLang)]:
            inputLang, outputLang = direction
            trainBatches = trainDataset.getLanguageBatches(inputLang, batchSize)

            for batch in trainBatches:
                batchData = trainDataset.collateFn(batch)
                inputTokens = batchData["tokens"].to(device)

                # input lang -> output lang
                outputHypothesis = model.forward(
                    srcTokens = inputTokens,
                    tgtTokens = None,
                    targetLang = outputLang,
                    mode = "translate"
                ).argmax(dim = -1)

                # output lang -> input lang
                reconstruction = model.forward(
                    srcTokens = outputHypothesis,
                    tgtTokens = inputTokens,
                    targetLang = inputLang,
                    mode = "translate"
                )

                loss = criterion(
                    reconstruction.view(-1, reconstruction.size(-1)),
                    inputTokens.view(-1),
                    ignore_index = padToken
                )

                with torch.no_grad():
                    preds = reconstruction.argmax(dim = -1)
                    correct = (preds == inputTokens) & (inputTokens != padToken)
                    accuracy = correct.sum().item() / (inputTokens != padToken).sum().item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = clipNorm)
                optimizer.step()

                if writer is not None:
                    writer.add_scalar(f"{inputLang}/train/loss", loss.item(), globalStep)
                    writer.add_scalar(f"{inputLang}/train/accuracy", accuracy, globalStep)
                    writer.add_scalar(f"{inputLang}/train/lr", optimizer.param_groups[0]["lr"], globalStep)

            globalStep += 1
        
        # validation
        model.eval()
        for direction in [(srcLang, tgtLang), (tgtLang, srcLang)]:
            inputLang, outputLang = direction
            valBatches = validDataset.getLanguageBatches(inputLang, batchSize)
            valLoss, valAcc, valTokens = 0.0, 0.0, 0

            with torch.no_grad():
                for batch in valBatches:
                    batchData = validDataset.collateFn(batch)
                    inputTokens = batchData["tokens"].to(device)

                    outputHypothesis = model.forward(
                        srcTokens = inputTokens,
                        tgtTokens = None,
                        targetLang = outputLang,
                        mode = "translate"
                    ).argmax(dim = -1)

                    reconstruction = model.forward(
                        srcTokens = outputHypothesis,
                        tgtTokens = inputTokens,
                        targetLang = inputLang,
                        mode = "translate"
                    )

                    loss = criterion(
                        reconstruction.view(-1, reconstruction.size(-1)),
                        inputTokens.view(-1),
                        ignore_index = padToken
                    )

                    preds = reconstruction.argmax(dim = -1)
                    correct = (preds == inputTokens) & (inputTokens != padToken)
                    valLoss += loss.item()
                    valAcc += correct.sum().item()
                    valTokens += (inputTokens != padToken).sum().item()
                
                if writer is not None:
                    writer.add_scalar(f"{inputLang}/val/loss", valLoss / valTokens, epoch)
                    writer.add_scalar(f"{inputLang}/val/accuracy", valAcc / valTokens, epoch)

                scheduler.step(valLoss / valTokens)
                logger.info(f"[{inputLang}] Val Loss: {valLoss / valTokens:.4f}, Val Acc: {valAcc / valTokens:.4f}")
            
        if (epoch + 1) % saveInterval == 0:
            torch.save(model.state_dict(), f"{checkpointDir}/backtranslation_epoch{epoch+1}.pt")

        model.train()

def startTrain(
    root: str,
    languages: Tuple[str, str],
    tokenizer: torch.nn.Module,
    modelConfig: ModelConfig,
    numEpochs: int,
    checkpointDir: str,
    shouldLog: bool,
    batchSize: int,
    saveInterval: int,
    autoencoderCheckpoint: Optional[str] = None
):
    
    trainDataset = BaseDataset(
        dataRoot = root,
        languages = languages,
        tokenizer = tokenizer,
        split = "train",
        name = f"{languages[0]}_{languages[1]}_train_dataset",
    )
    validDataset = BaseDataset(
        dataRoot = root,
        languages = languages,
        tokenizer = tokenizer,
        split = "valid",
        name = f"{languages[0]}_{languages[1]}_valid_dataset",
    )

    model = TextTransformerModel(modelConfig = modelConfig)

    if autoencoderCheckpoint and os.path.isfile(autoencoderCheckpoint):
        model.load_state_dict(torch.load(
            autoencoderCheckpoint,
            map_location = "cpu",
            weights_only = True
        ))
        logger.info(f"Loaded autoencoder checkpoint from {autoencoderCheckpoint}")
    else:
        logger.warn("Training from scratch for back-translation, performance will take a hit")
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = "min",
        factor = 0.5,
        patience = 2
    )

    clipNorm = 1.0
    writer = SummaryWriter(log_dir = "runs/backtranslation_phase2") if shouldLog else None

    padTokenIdx = modelConfig.padToken
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing = 0.1, 
        ignore_index = padTokenIdx
    )

    trainBackTranslationStage(
        model = model,
        trainDataset = trainDataset,
        validDataset = validDataset,
        optimizer = optimizer,
        criterion = criterion,
        batchSize = batchSize,
        scheduler = scheduler,
        clipNorm = clipNorm,
        saveInterval = saveInterval,
        writer = writer,
        checkpointDir = checkpointDir,
        numEpochs = numEpochs,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        padToken = padTokenIdx,
        tokenizer = tokenizer,
    )