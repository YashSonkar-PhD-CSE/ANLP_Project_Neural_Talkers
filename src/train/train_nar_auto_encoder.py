from typing import Tuple, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from tqdm import tqdm
from collections import Counter

from ..config import ModelConfig
from ..datasets import BaseDataset
from ..models.nar_model import NARTextTransformerModel
from ..utils import glanceInput, frequencyMaskInput

logging.basicConfig(filename="./nar_autoencoder_phase1_logs.txt")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def computeTokenFrequencies(dataset: BaseDataset, padToken: int) -> Counter:
    counter = Counter()
    for lang in dataset.languages:
        for batch in dataset.getLanguageBatches(lang, batchSize=64):
            for sample in batch:
                tokens = sample["tokens"]
                counter.update([t for t in tokens if t != padToken])
    return counter


def trainNARAutoEncoderStage(
    model: NARTextTransformerModel,
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
    tokenizer=None,
    glanceFraction: float = 0.5,
    freqMaskFraction: float = 0.3
):
    model.to(device)
    model.train()
    globalStep = 0

    logger.info("Computing token frequencies for frequency-based masking...")
    tokenFreqs = computeTokenFrequencies(trainDataset, padToken)

    for epoch in range(numEpochs):
        logger.info(f"Epoch: {epoch + 1}")
        for lang in trainDataset.languages:
            trainBatches = trainDataset.getLanguageBatches(lang, batchSize)
            for batch in tqdm(trainBatches, desc=f"NAR Train [{lang}] Epoch {epoch+1}", leave=False):
                batchData = trainDataset.collateFn(batch)
                original = batchData["tokens"].to(device)

                glanced = glanceInput(original, padToken=padToken, keepFraction=glanceFraction)
                masked = frequencyMaskInput(glanced, padToken=padToken, tokenFreqs=tokenFreqs, maskFraction=freqMaskFraction)

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
                for batch in tqdm(valBatches, desc=f"NAR Val [{lang}] Epoch {epoch+1}", leave=False):
                    batchData = validDataset.collateFn(batch)
                    original = batchData["tokens"].to(device)

                    glanced = glanceInput(original, padToken=padToken, keepFraction=glanceFraction)
                    masked = frequencyMaskInput(glanced, padToken=padToken, tokenFreqs=tokenFreqs, maskFraction=freqMaskFraction)

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
            glanced = glanceInput(original, padToken=padToken, keepFraction=glanceFraction)
            masked = frequencyMaskInput(glanced, padToken=padToken, tokenFreqs=tokenFreqs, maskFraction=freqMaskFraction)
            with torch.no_grad():
                output = model(srcTokens=masked, tgtTokens=None, targetLang=lang, mode="reconstruct")
                preds = output.argmax(dim=-1)

            if tokenizer and hasattr(tokenizer, "decode") and writer is not None:
                writer.add_text(f"{lang}/reconstruction/original", tokenizer.decode(original.squeeze().tolist()), epoch)
                writer.add_text(f"{lang}/reconstruction/masked", tokenizer.decode(masked.squeeze().tolist()), epoch)
                writer.add_text(f"{lang}/reconstruction/predicted", tokenizer.decode(preds.squeeze().tolist()), epoch)

        if (epoch + 1) % saveInterval == 0:
            torch.save(model.state_dict(), f"{checkpointDir}/nar_autoencoder_epoch{epoch+1}.pt")

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
    import os
    from torch.utils.tensorboard import SummaryWriter
    from ..datasets import BaseDataset
    from ..models.nar_model import NARTextTransformerModel
    from ..train_nar_autoencoder import trainNARAutoEncoderStage

    os.makedirs(checkpointDir, exist_ok=True)

    trainDataset = BaseDataset(
        dataRoot=root,
        languages=languages,
        tokenizer=tokenizer,
        split="train",
        name=f"{languages[0]}_{languages[1]}_train_dataset",
    )
    validDataset = BaseDataset(
        dataRoot=root,
        languages=languages,
        tokenizer=tokenizer,
        split="valid",
        name=f"{languages[0]}_{languages[1]}_valid_dataset",
    )

    model = NARTextTransformerModel(modelConfig=modelConfig)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=numEpochs // 2, T_mult=1, eta_min=1e-5
    )
    clipNorm = 1.0

    writer = SummaryWriter(log_dir="runs/nar_autoencoder_phase1") if shouldLog else None

    padTokenIdx = modelConfig.padToken
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=padTokenIdx,
        reduction="mean"
    )

    trainNARAutoEncoderStage(
        model=model,
        trainDataset=trainDataset,
        validDataset=validDataset,
        optimizer=optimizer,
        criterion=criterion,
        batchSize=batchSize,
        scheduler=scheduler,
        clipNorm=clipNorm,
        saveInterval=saveInterval,
        writer=writer,
        checkpointDir=checkpointDir,
        numEpochs=numEpochs,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        padToken=padTokenIdx,
        tokenizer=tokenizer,
        glanceFraction=0.5,
        freqMaskFraction=0.3
    )
