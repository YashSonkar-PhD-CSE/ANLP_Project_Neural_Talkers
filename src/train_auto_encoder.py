from typing import Tuple, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from tqdm import tqdm

from .config import ModelConfig
from .datasets import BaseDataset
from .model import TextTransformerModel
from .utils import maskInput

logging.basicConfig(filename = "./autoencoder_phase1_logs.txt")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def trainAutoEncoderStage(
    model: TextTransformerModel,
    trainDataset: BaseDataset,
    validDataset: BaseDataset,
    optimizer: torch.optim.Optimizer,
    writer: Optional[SummaryWriter],
    batchSize: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    clipNorm: float,
    saveInterval: int = 1,
    checkpointDir: str = "./checkpoints/",
    criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    numEpochs: int = 10,
    padToken: int = 0,
    device: torch.device = torch.device("cpu"),
    tokenizer = None
):
    model.to(device)
    model.train()
    globalStep = 0

    for epoch in range(numEpochs):
        logger.info(f"Epoch: {epoch + 1}")
        for lang in trainDataset.languages:
            trainBatches = trainDataset.getLanguageBatches(lang, batchSize)
            for batch in tqdm(trainBatches, desc=f"Train [{lang}] Epoch {epoch+1}", leave=False):
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
                for batch in tqdm(valBatches, desc=f"Val [{lang}] Epoch {epoch+1}", leave=False):
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
                    )

                    preds = output.argmax(dim=-1)
                    correct = (preds == original) & (original != padToken)
                    valLoss += loss.item()
                    valAcc += correct.sum().item()
                    valTokens += (original != padToken).sum().item()
            if writer is not None:
                writer.add_scalar(f"{lang}/val/loss", valLoss / valTokens, epoch)
                writer.add_scalar(f"{lang}/val/accuracy", valAcc / valTokens, epoch)
            scheduler.step()
            logger.info(f"[{lang}] Val Loss: {valLoss / valTokens:.4f}, Val Acc: {valAcc / valTokens:.4f}")

            # Log sample reconstruction
            sample = validDataset.getRandomSample(lang)
            original = sample.tokenIds.unsqueeze(0).to(device)
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
    root: str,
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
        dataRoot = root,
        languages = languages,
        tokenizer = tokenizer,
        split = "test",
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

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs, eta_min=1e-6)
    clipNorm = 1.0
    writer = None
    if shouldLog:
        writer = SummaryWriter(log_dir = "runs/autoencoder_phase1")

    padTokenIdx = modelConfig.padToken
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
        tokenizer = tokenizer
    )
