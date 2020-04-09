<div align="center">

[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)](https://github.com/catalyst-team/catalyst)

**Accelerated DL R&D**

[![Build Status](http://66.248.205.49:8111/app/rest/builds/buildType:id:Catalyst_Deploy/statusIcon.svg)](http://66.248.205.49:8111/project.html?projectId=Catalyst&tab=projectOverview&guest=1)
[![CodeFactor](https://www.codefactor.io/repository/github/catalyst-team/catalyst/badge)](https://www.codefactor.io/repository/github/catalyst-team/catalyst)
[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)
[![PyPI Status](https://pepy.tech/badge/catalyst)](https://pepy.tech/project/catalyst)

[![Twitter](https://img.shields.io/badge/news-on%20twitter-499feb)](https://twitter.com/catalyst_core)
[![Telegram](https://img.shields.io/badge/channel-on%20telegram-blue)](https://t.me/catalyst_team)
[![Slack](https://img.shields.io/badge/Catalyst-slack-success)](https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw)
[![Github contributors](https://img.shields.io/github/contributors/catalyst-team/catalyst.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/catalyst/graphs/contributors)

</div>

PyTorch framework for Deep Learning research and development.
It was developed with a focus on reproducibility,
fast experimentation and code/ideas reusing.
Being able to research/develop something new,
rather than write another regular train loop. <br/>
Break the cycle - use the Catalyst!

Project [manifest](https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md). Part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/). Part of [Catalyst Ecosystem](https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing):
- [Alchemy](https://github.com/catalyst-team/alchemy) - Experiments logging & visualization
- [Catalyst](https://github.com/catalyst-team/catalyst) - Accelerated Deep Learning Research and Development
- [Reaction](https://github.com/catalyst-team/reaction) - Convenient Deep Learning models serving

[Catalyst at AI Landscape](https://landscape.lfai.foundation/selected=catalyst).

---

# Catalyst.Detection [![Build Status](https://travis-ci.com/catalyst-team/detection.svg?branch=master)](https://travis-ci.com/catalyst-team/detection) [![Github contributors](https://img.shields.io/github/contributors/catalyst-team/detection.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/detection/graphs/contributors)

Based on [Objects as points](https://arxiv.org/abs/1904.07850) article by [Xingyi Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+X), [Dequan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+D), [Philipp Krähenbühl](https://arxiv.org/search/cs?searchtype=author&query=Kr%C3%A4henb%C3%BChl%2C+P)

### Training in your dataset
0. Install requirements ```pip install -r requirements.txt```

1. Copy all images to one directory or two different directories for train and validation.

1. Create ```markup_train.json``` as json file in MSCOCO format using ```COCODetectionFactory``` from ```data_preparation.py```. This class may be copied to your dataset generator. See documentation in code comments.  If your dataset are already in this format, go to next step.

1. Specify perameters and in ```config/centernet_detection_config.yml```.

1. Run catalyst ```catalyst-dl run --config=./configs/centernet_detection_config.yml```

1. When you change dataset, you must delete cache files ```markup_*.json.cache``` because this files contain preprocessed bounding boxes info.
