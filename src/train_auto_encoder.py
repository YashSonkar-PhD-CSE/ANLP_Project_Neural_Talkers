from typing import Tuple, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import os

from .config import ModelConfig
from .datasets import BaseDataset
from .model import TextTransformerModel
from .utils import maskInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def trainAutoEncoderStage(
    model: TextTransformerModel,
    trainDataset: BaseDataset,
    validDataset: BaseDataset,
    optimizer: torch.optim.Optimizer,
    writer: Optional[SummaryWriter],
    batchSize: int,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    clipNorm: float,
    saveInterval: int = 1,
    checkpointDir: str = "./checkpoints/",
    criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    numEpochs: int = 10,
    padToken: int = 0,
    device: torch.device = torch.device("cpu"),
    tokenizer = None  # Optional decode method
):
    model.to(device)
    model.train()
    globalStep = 0

    for epoch in range(numEpochs):
        logger.info(f"Epoch: {epoch + 1}")
        for lang in trainDataset.languages:
            trainBatches = trainDataset.getLanguageBatches(lang, batchSize)
            for batch in trainBatches:
                batchData = trainDataset.collateFn(batch)
                original = batchData["tokens"].to(device)
                masked = maskInput(original, padToken=padToken)

                optimizer.zero_grad()
                output = model.forward(
                    srcTokens=masked,
                    tgtTokens=None,
                    targetLang=lang,
                    mode="reconstruct"
                )

                loss = criterion(
                    output.view(-1, output.size(-1)),
                    original.view(-1),
                    ignore_index=padToken
                )

                with torch.no_grad():
                    preds = output.argmax(dim=-1)
                    correct = (preds == original) & (original != padToken)
                    accuracy = correct.sum().item() / (original != padToken).sum().item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipNorm)
                optimizer.step()
                if writer is not None:
                    writer.add_scalar(f"{lang}/train/loss", loss.item(), globalStep)
                    writer.add_scalar(f"{lang}/train/accuracy", accuracy, globalStep)
                    writer.add_scalar(f"{lang}/train/lr", optimizer.param_groups[0]["lr"], globalStep)
                globalStep += 1

        # Validation loop
        model.eval()
        for lang in validDataset.languages:
            valBatches = validDataset.getLanguageBatches(lang, batchSize)
            valLoss, valAcc, valTokens = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in valBatches:
                    batchData = validDataset.collateFn(batch)
                    original = batchData["tokens"].to(device)
                    masked = maskInput(original, padToken=padToken)

                    output = model.forward(
                        srcTokens=masked,
                        tgtTokens=None,
                        targetLang=lang,
                        mode="reconstruct"
                    )

                    loss = criterion(
                        output.view(-1, output.size(-1)),
                        original.view(-1),
                        ignore_index=padToken
                    )

                    preds = output.argmax(dim=-1)
                    correct = (preds == original) & (original != padToken)
                    valLoss += loss.item()
                    valAcc += correct.sum().item()
                    valTokens += (original != padToken).sum().item()
            if writer is not None:
                writer.add_scalar(f"{lang}/val/loss", valLoss / valTokens, epoch)
                writer.add_scalar(f"{lang}/val/accuracy", valAcc / valTokens, epoch)
            scheduler.step(metrics=valLoss / valTokens)
            logger.info(f"[{lang}] Val Loss: {valLoss / valTokens:.4f}, Val Acc: {valAcc / valTokens:.4f}")

            # Log sample reconstruction
            sample = validDataset.getRandomSample(lang)
            original = sample.text.unsqueeze(0).to(device)
            masked = maskInput(original, padToken=padToken)
            with torch.no_grad():
                output = model(srcTokens=masked, tgtTokens=None, targetLang=lang, mode="reconstruct")
                preds = output.argmax(dim=-1)

            if tokenizer and hasattr(tokenizer, "decode") and writer is not None:
                writer.add_text(f"{lang}/reconstruction/original", tokenizer.decode(original.squeeze().tolist()), epoch)
                writer.add_text(f"{lang}/reconstruction/masked", tokenizer.decode(masked.squeeze().tolist()), epoch)
                writer.add_text(f"{lang}/reconstruction/predicted", tokenizer.decode(preds.squeeze().tolist()), epoch)

        if (epoch + 1) % saveInterval == 0:
            torch.save(model.state_dict(), f"{checkpointDir}/autoencoder_epoch{epoch+1}.pt")

        model.train()

def startTrain(
    languages: Tuple[str, str],
    tokenizer: torch.nn.Module,
    modelConfig: ModelConfig,
    numEpochs: int,
    checkpointDir: str,
    shouldLog: bool,
    batchSize: int,
    saveInterval: int
):
    os.makedirs(checkpointDir, exist_ok = True)
    trainDataset = BaseDataset(
        languages = languages,
        tokenizer = tokenizer,
        split = "train",
        name = f"{languages[0]}_{languages[1]}_train_dataset",
    )
    validDataset = BaseDataset(
        languages = languages,
        tokenizer = tokenizer,
        split = "valid",
        name = f"{languages[0]}_{languages[1]}_valid_dataset",
    )

    model = TextTransformerModel(modelConfig = modelConfig)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor = 0.5, patience = 2)
    clipNorm = 1.0
    writer = None
    if shouldLog:
        writer = SummaryWriter(log_dir = "runs/autoencoder_phase1")

    padTokenIdx = 0
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing = 0.1,
        ignore_index = padTokenIdx,
    )

    trainAutoEncoderStage(
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
    )