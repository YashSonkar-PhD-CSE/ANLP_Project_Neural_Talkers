import argparse
from typing import Any, Tuple
from datasets import load_dataset
import os

def corpusToRepoId(corpusName: str) -> Any:
    if corpusName == "en_hi":
        return ("ai4bharat/samanantar", "hi")
    elif corpusName == "en_la":
        return "grosenthal/latin_english_translation"
    
def createDirs(*args):
    for folder in args:
        assert isinstance(folder, str)
        os.makedirs(folder, exist_ok=True)

def saveSamplesFromDs(dataset, path: str, numSamples: int, srcKey: str, tgtKey: str, languages: Tuple[str, str]):
    it = iter(dataset)
    dirs = [os.path.join(path, lang) for lang in languages]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        
    for i in range(numSamples):
        print(f"Progress: {i} / {numSamples} ({i * 100 / numSamples:.2f}%)", " " * 30, end = "\r")
        sample = next(it)
        for key, name, dir in zip((srcKey, tgtKey), languages, dirs):
            text = sample[key]
            with open(os.path.join(dir, f"{i}.txt"), "w", encoding='utf-8') as txtFile:
                txtFile.write(text.strip())
    
def downloadCorpus(corpus: str, numTrainSamples: int = 100_000, numTestSamples: int = 500):
    repoId = corpusToRepoId(corpus)
    print(f"Downloading {corpus} corpus from repo {repoId}")
    outDir = f"./{corpus}"
    trainDir = os.path.join(outDir, "train")
    validDir = os.path.join(outDir, "valid")
    testDir = os.path.join(outDir, "test")

    createDirs(outDir, trainDir, validDir, testDir)
    if corpus == "en_la":
        assert isinstance(repoId, str)
        trainDs = load_dataset(repoId, split="train", streaming=True)
        testDs = load_dataset(repoId, split="test", streaming=True)
    elif corpus == "en_hi":
        assert isinstance(repoId, Tuple)
        ds = load_dataset(path=repoId[0], name=repoId[1], split="train", streaming=True)
        print("Split = train")
        saveSamplesFromDs(
            dataset = ds,
            path = trainDir,
            numSamples = numTrainSamples,
            srcKey = "src",
            tgtKey = "tgt",
            languages = ("en", "hi")
        )
        print("Split = valid")
        saveSamplesFromDs(
            dataset = ds,
            path = validDir,
            numSamples = int(numTrainSamples * 0.05),
            srcKey = "src",
            tgtKey = "tgt",
            languages = ("en", "hi")
        )
        print("Split = test")
        saveSamplesFromDs(
            dataset = ds,
            path = testDir,
            numSamples = numTestSamples,
            srcKey = "src",
            tgtKey = "tgt",
            languages = ("en", "hi")
        )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        choices=["en_hi", "en_la"],
        default = "en_hi"
    )
    parser.add_argument(
        "--num-train-samples",
        type = int,
        default = 100_000,
    )
    parser.add_argument(
        '--num-test-samples',
        type = int,
        default = 500
    )
    args = parser.parse_args()
    downloadCorpus(
        args.corpus,
        numTrainSamples = args.num_train_samples,
        numTestSamples = args.num_test_samples
    )
if __name__ == "__main__":
    main()