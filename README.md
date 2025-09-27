# Multi-lingual word embeddings without parallel data

## Table of contents
<ol>
    <li>Model architecture</li>
    <li>Training stages</li>
    <!-- <li>Important files</li> -->
    <li>How to train the model</li>
    <li>TODO Section</li>
</ol>

## Model Architecture
The model is a standard auto-regressive transformer with a unified encoder for multiple languages and multiple language-specific decoders. 
Currently, we only provide base model config with following specifications:
|Field|Value|
|----|----|
|Encoder layers|4|
|Decoder layers|4|
|Number of attention heads|8|
|Embedding dimension|512|
|maximum seq. len| 5000|
|Position encoding|RoPE|


Training is only done with RoPE but other variants can be trained by changing the config (code implementation is present).

## Training Stages:
Due to our assumption of absence of paired data, training is not straightforward and is carried out in two stages:

### Stage 1: Denoising Auto-Encoder:
In this, the model is trained for reconstructing source text (in same language). 20% of input tokens are randomly masked. Within one epoch, first encoder and one deocder is trained for the corresponding language is trained, then the decoder is swapped for that of the other language. This is repeated for n epochs untill the model learns to reconstruct text. 

This phase enables the encoder to learn robust representations for both languages.

### Stage 2: back-translation:
After auto-encoder training, the model is trained for translation task.
Similar to previous phase, there are two parts within each epoch. In the first phase, text in 1 language is translated to the other language by swapping the decoder and then translated back into the source language. Reconstruction loss is applied on the generated text and source text. The same procedure is applied for text in target language. This is repeated for n epochs.

## How to train the model
`train.py` is the entry-pont for training code in this codebase.

Sample command to train auto-encoder:
```bash
python train.py --train-phase autoencoder \
    --src-language en --tgt-language fr \
    --checkpoint-path ./checkpoints --model-config base \
    --batch-size 32 --save-interval 10 --log
```

Sampl command to train back-translation
```bash
python train.py --train-phase backtranslation \
    --src-language en --tgt-language fr \
    --checkpoint-path ./checkpoints --model-config base \
    --batch-size 32 --save-interval 10 --log \
    --autoencoder-checkpoint ./checkpoints/autoencoder_epoch1.pt
```