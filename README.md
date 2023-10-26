# asr
dl-audio homework. Описание репозитория:
* Папка `checkpoints` - сюда складывать итоговый чекпоинт и соответсвующий ему `config.json` (ниже написал что и как).
* Папка `hw_asr` - тут лежит основной код домашки по сути почти полностью это предоставленный шаблон, отдельно только можно прокомментировать модуль `augmentations`, там реализовал FrequentMasking, TimeMasking, TimeStretch, Gain, GaussianNoise, PitchShift, SpeedPerturbation, все работают в итоговом ране использовал только PitchShift и SpeedPerturbation (подробнее смотри в отчете), а также модуль `text_encoder`, там добавил возможность использовать бимсерч с лмкой, а также возможность BPE токенизации, но чтобы BPE работал надо закинуть в корневую директорию train.txt на котором BPE будет обучаться (у меня это текста из train-clean-100), там еще папка `language_models` куда скачиваются языковые модели, а также `tokenizers` сюда SentencePiece сохранит BPE токенайзер.
* `notebooks` - я обучался на кагле, тут ноутбук с которым я запускался на кагле (свои ключи затёр).
* `evaluate.py` - код чтобы посчитать скор по итоговым прогнозам.
* `test_config.json` - конфиг который 
* `train.txt` - тот самый файл, который нужен для BPE.

## Какие бонусы сделал:
1. Добавил бимсерч с ЛМкой, для этого использовал библиотеку pyctcdecode и kenlm, так как в экспериментах надо было считать метрики и по бимсерчу с языковой моделью и по бимсерчу без языковой модели, то в `hw_asr/text_encoder/ctc_char_text_encoder.py` реализовано два метода, один вызывает бимсерч с лмкой, другой без.
2. Добавил возможность токенизации bpe, реализовано в `hw_asr/text_encoder/char_text_encoder.py` через SentencePiece.

## Установка пакетов
Есть requirements.txt но он сделан по окружению с моего компьютера и при его использовании, например в kaggle ноутбуках возникают ошибки, поэтому если проект запускается на kaggle я советую не пытаться ставить requirements.txt, а поставить следуюшие библиотеки вручную:
```bash
pip install editdistance
pip install torch-audiomentations
pip install pyctcdecode
pip install speechbrain
pip install https://github.com/kpu/kenlm/archive/master.zip#sha256=4d002dcde70b52d519cafff4dc0008696c40cff1c9184a531b40c7b45905be6b
```

## Install LMs
Далее нужно скачать используемую языковую модель и распаковать ее куда надо.
```bash
wget https://us.openslr.org/resources/11/3-gram.arpa.gz
gunzip -c 3-gram.arpa.gz > hw_asr/text_encoder/language_models/3-gram.arpa
```

## Как получить результаты
Обучаем модель
```bash
python train.py --config hw_asr/configs/final_config.json
```
Скачайте веса и config.json по [ссылке](https://drive.google.com/drive/folders/1dEm-8cqA6d6om0WPiSTNRubdEBy2Cwt7?usp=drive_link) и поместите в папку `checkpoints`
```bash
python test.py --config hw_asr/configs/test_config.json --resume checkpoints/checkpoint-epoch40.pth
```
инференс вернет файл `output.json` и теперь мы хотим на нем посчитать метрики:
```bash
python evaluate.py --predictions output.json
```
и все, мы получили результаты.

Отчет об экспериментах смотреть тут: 
