import torch
from torch.utils.data import DataLoader
from geomloss import SamplesLoss
import random
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.evaluation.evaluation_dataset import EvaluationDataset
from src.models.ar_model import TextTransformerModel
from src.tokenizers import TokenizerModule

def evaluateEmbeddings(
    model: TextTransformerModel,
    tokenizer: TokenizerModule,
    outputDir: str,
    batchSize: int = 1,
    padTokenId: int = 0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    os.makedirs(outputDir, exist_ok=True)

    languages = tuple(model.decoder.keys())
    assert len(languages) == 2
    lang1, lang2 = languages

    encoder = model.encoder.to(device)
    encoder.eval()

    dataset = EvaluationDataset(
        languages=languages,
        split="test",
        tokenizer=tokenizer,
        dataRoot = "./data/en_la/"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        collate_fn=lambda batch: dataset.collateFn(batch, padTokenIdx=padTokenId)
    )

    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)
    pairedDistances = []
    unpairedDistances = []

    # Preload tokenized samples for unpaired access
    allSamples = [dataset[i] for i in range(len(dataset))]

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating Sinkhorn distances")):
        tokensLang1 = batch["tokens"][lang1].to(device)
        tokensLang2 = batch["tokens"][lang2].to(device)

        with torch.no_grad():
            embLang1 = encoder(tokensLang1)[0]  # [N, D]
            embLang2 = encoder(tokensLang2)[0]  # [N, D]
            pairedDistances.append(sinkhorn(embLang1, embLang2).item())

            for _ in range(5):
                j = random.choice([idx for idx in range(len(allSamples)) if idx != i])
                unpairedTokens = allSamples[j].tokenIds[lang2].unsqueeze(0).to(device)
                embUnpaired = encoder(unpairedTokens)[0]
                unpairedDistances.append(sinkhorn(embLang1, embUnpaired).item())

    avgPaired = sum(pairedDistances) / len(pairedDistances)
    avgUnpaired = sum(unpairedDistances) / len(unpairedDistances)

    print(f"Average paired Sinkhorn distance: {avgPaired:.4f}")
    print(f"Average unpaired Sinkhorn distance: {avgUnpaired:.4f}")

    # Save results
    result = {
        "avgPaired": avgPaired,
        "avgUnpaired": avgUnpaired,
        "pairedDistances": pairedDistances,
        "unpairedDistances": unpairedDistances
    }

    outputPath = os.path.join(outputDir, "sinkhorn_distances.json")
    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {outputPath}")

    # Optional: visualize
    plt.hist(pairedDistances, bins=50, alpha=0.6, label="Paired")
    plt.hist(unpairedDistances, bins=50, alpha=0.6, label="Unpaired")
    plt.legend()
    plt.title("Sinkhorn Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, f"sinkhorn_histogram_{lang1}_{lang2}.png"))
    plt.close()

    return result
