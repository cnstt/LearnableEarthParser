# LearnableEarthParser

> [!IMPORTANT]  
> This repository is a fork of the original repo [LearnableEarthParser](https://github.com/romainloiseau/LearnableEarthParser). For more informations, check out the original repository. Special thanks to [@romainloiseau](https://github.com/romainloiseau) for his great work!
>
> This fork adds specific changes detailed below, and is dedicated to work in the combination with this special version of [EarthParserDataset](https://github.com/cnstt/EarthParserDataset) and the 3D Lidar scenes of [genBoxes](https://github.com/cnstt/genBoxes).

## Installation

For reproducibility purposes, you can set up and use the following environment for LearnableEarthParser:
```
conda env create -f environment_work.yml
conda activate lep-original
```

## Main changes

### Masking approaches

Due to the caracteristics of Lidar sampling, especially in the context of autonomous driving, the scenes can often be highly occluded.
Therefore, to obtain meaningful prototypes, the strategy explored here consists of applying different masking techniques in one crucial part of the reconstruction loss: [compute_l_PX](learnableearthparser/model/ours.py#L213), or $L_{acc}$ in the [paper](https://arxiv.org/abs/2304.09704).

The following masking strategies have been implemented in [learnableearthparser/model/ours.py](learnableearthparser/model/ours.py) :
- [clip_below_threshold](learnableearthparser/model/ours.py#L44) ;
- [mask_self_occultation](learnableearthparser/model/ours.py#L82) ;
- [mask_general_occultation](learnableearthparser/model/ours.py#L151) .

### Other important changes

- New curriculum options.
- Added logging of new numerical values to Tensorboard.
- Added **visualisation of Lidar position and masking in Tensorboard**.
- Option to **force ground truth positions of prototypes during training**.
- New prototype shapes.
- First **mesh prototypes** implementation.
- **Video rendering** [script](learnableearthparser/utils/render_video.py) to show the **training process**.
    <details>
    <summary>For this script, a special conda environment is required.</summary>

     ```
    conda env create -f env_render_video.yml
    conda activate render
    ```
    </details>
- New **config files for different experiment setups**. You can for example have a look at one here: [configs/experiment/5complex_light_boxProto_velod_allScene_SINGLE.yaml](configs/experiment/5complex_light_boxProto_velod_allScene_SINGLE.yaml). It **makes use of the new options listed above**.
