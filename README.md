# asr
dl-audio homework. More details located in wanbd [report](https://wandb.ai/dzhunkoffski/asr_project/reports/Report--Vmlldzo2NTc3OTEw)
```bash
asr
├── checkpoints         <-- move here pretrained model checkpoint and corresponding `config.json`.
├── hw_asr              <-- module with model, trainer, dataloader and etc.
├── notebooks           <-- example notebooks 
    └── train_kaggle_pipeline.ipynb     <-- example how to use this in kaggle.
```

## Install packages
```bash
pip install editdistance
pip install torch-audiomentations
pip install pyctcdecode
pip install speechbrain
pip install https://github.com/kpu/kenlm/archive/master.zip#sha256=4d002dcde70b52d519cafff4dc0008696c40cff1c9184a531b40c7b45905be6b
```

## Install LMs
```bash
wget https://us.openslr.org/resources/11/3-gram.arpa.gz
gunzip -c 3-gram.arpa.gz > hw_asr/text_encoder/language_models/3-gram.arpa
```

## Train and inference:
Train model:
```bash
python train.py --config hw_asr/configs/final_config.json
```
Download checkpoint and config.json from the [link](https://drive.google.com/drive/folders/1dEm-8cqA6d6om0WPiSTNRubdEBy2Cwt7?usp=drive_link) and move them to `checkpoints`.

Now, infere pretrained model:
```bash
python test.py --config hw_asr/configs/test_config.json --resume checkpoints/checkpoint-epoch40.pth
```
You will get `output.json` with predicions. The following command will evaluate model predictions:
```bash
python evaluate.py --predictions output.json
```
More details on model and training can be found here (TODO WANDB REPORT link).
