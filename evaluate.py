import os
import torch

from src.models.ar_model import TextTransformerModel
from src.utils import EvaluationArgs, makeEvaluationParser, getModelConfigFromStateDict
from src.evaluation.embedding_evaluation import evaluateEmbeddings 

def main():
    parser = makeEvaluationParser()
    args: EvaluationArgs = parser.parse_args()  # type: ignore

    assert os.path.isfile(args.checkpoint_path) and args.checkpoint_path.endswith(".pt"), \
        f"Invalid checkpoint path ({args.checkpoint_path})"

    assert args.output_dir is not None, "Output directory must be specified for evaluation results"

    stDict = torch.load(args.checkpoint_path, map_location='cpu')
    modelConfig, tokenizer = getModelConfigFromStateDict(
        stDict,
        peType="sinusoidal",
        tokenizerType="bpe"
    )
    model = TextTransformerModel(modelConfig=modelConfig)
    model.load_state_dict(stDict)

    if args.evaluation_type == "translation":
        pass  # TODO: Invoke method to evaluate translations
    elif args.evaluation_type == "embedding":
        evaluateEmbeddings(
            model=model,
            tokenizer=tokenizer,
            outputDir=args.output_dir,
            batchSize=4,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        raise NotImplementedError(f"Evaluation type: {args.evaluation_type} not implemented yet")

if __name__ == "__main__":
    main()
