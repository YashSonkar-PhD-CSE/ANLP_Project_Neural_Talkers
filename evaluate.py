import os
import torch

from src.models.ar_model import TextTransformerModel
from src.utils import EvaluationArgs, makeEvaluationParser, getModelConfigFromStateDict

def main():
    parser = makeEvaluationParser()
    args: EvaluationArgs = parser.parse_args() # type: ignore

    assert os.path.isfile(args.checkpoint_path) and args.checkpoint_path.endswith(".pt"), f"Invalid checkpoint path ({args.checkpoint_path})"

    stDict = torch.load(args.checkpoint_path, map_location='cpu')
    modelConfig = getModelConfigFromStateDict(stDict)
    model = TextTransformerModel()

    if args.evaluation_type == "translation":
        pass # TODO: Invoke method to evaluation translations
    elif args.evaluation_type == "embedding":
        pass # TODO: Invoke embedding evaluation method
    else:
        raise NotImplementedError(f"Evluation type: {args.evaluation_type} not implemented yet")
    

if __name__ == "__main__":
    main()