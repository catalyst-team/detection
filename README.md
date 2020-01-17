# Detection catalyst pipeline

Based on [Objects as points](https://arxiv.org/abs/1904.07850) article by [Xingyi Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+X), [Dequan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+D), [Philipp Krähenbühl](https://arxiv.org/search/cs?searchtype=author&query=Kr%C3%A4henb%C3%BChl%2C+P)

### Training in your dataset
0. Install requirements ```pip install -r requirements.txt```

1. Copy all images to one directory or two different directories for train and validation.

1. Create ```markup_train.json``` as json file in MSCOCO format using ```COCODetectionFactory``` from ```data_preparation.py```. This class may be copied to your dataset generator. See documentation in code comments.  If your dataset are already in this format, go to next step.

1. Specify perameters and in ```config/centernet_detection_config.yml```.

1. Run catalyst ```catalyst-dl run --config=./configs/centernet_detection_config.yml```

1. When you change dataset, you must delete cache files ```markup_*.json.cache``` because this files contain preprocessed bounding boxes info.