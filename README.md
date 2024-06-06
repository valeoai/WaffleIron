# WaffleIron

![](./illustration.png)

[**Using a Waffle Iron for Automotive Point Cloud Semantic Segmentation**](http://arxiv.org/abs/2301.10100)  
[*Gilles Puy*<sup>1</sup>](https://sites.google.com/site/puygilles/home),
[*Alexandre Boulch*<sup>1</sup>](http://boulch.eu),
[*Renaud Marlet*<sup>1,2</sup>](http://imagine.enpc.fr/~marletr/)  
<sup>1</sup>*valeo.ai, France* and <sup>2</sup>*LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, France*.

If you find this code or work useful, please cite the following [paper](http://arxiv.org/abs/2301.10100):
```
@inproceedings{puy23waffleiron,
  title={Using a Waffle Iron for Automotive Point Cloud Semantic Segmentation},
  author={Puy, Gilles and Boulch, Alexandre and Marlet, Renaud},
  booktitle={ICCV},
  year={2023}
}
```

## Main updates

- **[Jun. 06, 2024]** The code is now compatible with `pytorch>=2.0` thanks to a new implementation of the 3D to 2D projection. 
- **[Sep. 21, 2023]** This work was accepted at ICCV23. The code and trained models were updated on September 21, 2023 to allow reproduction of the scores in the published version. If you need to access the preliminary trained models you can refer to this [section](#Preliminary-version). Note that those preliminary models are not performing as well as those used in the published version.


## Installation

We use the following environment:
```
conda create -n waffleiron
conda activate waffleiron
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install pyaml==23.12.0 tqdm==4.63.0 scipy==1.8.0 tensorboard==2.16.2
git clone https://github.com/valeoai/WaffleIron
cd WaffleIron
pip install -e ./
```

Alternatively, the code was updated on June 6, 2024 to make it compatible with `pytorch>=2.0`. You should able to use the following environment. In case of problem with this environment, please inform us by reporting an issue.
```
conda create -n waffleiron
conda activate waffleiron
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyaml==23.12.0 tqdm==4.63.0 scipy==1.13.1 tensorboard==2.16.2
git clone https://github.com/valeoai/WaffleIron
cd WaffleIron
pip install -e ./
```

Download the trained models:
```
wget https://github.com/valeoai/WaffleIron/files/10294733/info_datasets.tar.gz
tar -xvzf info_datasets.tar.gz
wget https://github.com/valeoai/WaffleIron/releases/download/v0.2.0/waffleiron_nuscenes.tar.gz
tar -xvzf waffleiron_nuscenes.tar.gz
wget https://github.com/valeoai/WaffleIron/releases/download/v0.2.0/waffleiron_kitti.tar.gz
tar -xvzf waffleiron_kitti.tar.gz
```

If you want to uninstall this package, type `pip uninstall waffleiron`.


## Testing pretrained models

### Option 1: Using this code

To evaluate the nuScenes trained model, type
```
python launch_train.py \
--dataset nuscenes \
--path_dataset /path/to/nuscenes/ \
--log_path ./pretrained_models/WaffleIron-48-384__nuscenes/ \
--config ./configs/WaffleIron-48-384__nuscenes.yaml \
--fp16 \
--multiprocessing-distributed \
--restart \
--eval
```
This should give you a final mIoU of 77.6%.

**Remark**: *If your model was trained on one gpu with the argument `--gpu 0`, replace `--multiprocessing-distributed` with `--gpu 0` for evaluation with the above command.*

To evaluate the SemanticKITTI trained model, type
```
python launch_train.py \
--dataset semantic_kitti \
--path_dataset /path/to/kitti/ \
--log_path ./pretrained_models/WaffleIron-48-256__kitti/ \
--config ./configs/WaffleIron-48-256__kitti.yaml \
--fp16 \
--multiprocessing-distributed \
--restart \
--eval
```
This should give you a final mIoU of 68.0%.

**Remark:** *On SemanticKITTI, the code above will extract object instances on the train set (despite this being not necessary for validation) because this augmentation is activated for training on this dataset (and this code re-use the training script). This can be bypassed by editing the `yaml` config file and changing the entry `instance_cutmix` to `False`. The instances are saved automatically in `/tmp/semantic_kitti_instances/`.*


### Option 2: Using the official APIs

The second option writes the predictions on disk and the results can be computed using the official nuScenes or SemanticKITTI APIs. This option also allows you to perform test time augmentations, which is not possible with Option 1 above. These scripts should be useable for submission of the official benchmarks.

#### nuScenes

To extract the prediction with the model trained on nuScenes, type
```
python eval_nuscenes.py \
--path_dataset /path/to/nuscenes/ \
--config ./configs/WaffleIron-48-384__nuscenes.yaml \
--ckpt ./pretrained_models/WaffleIron-48-384__nuscenes/ckpt_last.pth \
--result_folder ./predictions_nuscenes \
--phase val \
--num_workers 12
```
or, if you want to use, e.g., 10 votes with test time augmentations,
```
python eval_nuscenes.py \
--path_dataset /path/to/nuscenes/ \
--config ./configs/WaffleIron-48-384__nuscenes.yaml \
--ckpt ./pretrained_models/WaffleIron-48-384__nuscenes/ckpt_last.pth \
--result_folder ./predictions_nuscenes \
--phase val \
--num_workers 12 \
--num_votes 10 \
--batch_size 10
```
You can reduce `batch_size` to 5, 2 or 1 depending on the available memory.

These predictions can be evaluated using the official nuScenes API as follows
```
git clone https://github.com/nutonomy/nuscenes-devkit.git
python nuscenes-devkit/python-sdk/nuscenes/eval/lidarseg/evaluate.py \
--result_path ./predictions_nuscenes \
--eval_set val \
--version v1.0-trainval \
--dataroot /path/to/nuscenes/ \
--verbose True  
```

#### SemanticKITTI

To extract the prediction with the model trained on SemanticKITTI, type
```
python eval_kitti.py \
--path_dataset /path/to/kitti/ \
--ckpt ./pretrained_models/WaffleIron-48-256__kitti/ckpt_last.pth \
--config ./configs/WaffleIron-48-256__kitti.yaml \
--result_folder ./predictions_kitti \
--phase val \
--num_workers 12
```

The predictions can be evaluated using the official APIs by typing
```
git clone https://github.com/PRBonn/semantic-kitti-api.git
cd semantic-kitti-api/
python evaluate_semantics.py \
--dataset /path/to/kitti//dataset \
--predictions ../predictions_kitti \
--split valid
```

## Training

### nuScenes

To retrain the WaffleIron-48-384 backbone on nuScenes type
```
python launch_train.py \
--dataset nuscenes \
--path_dataset /path/to/nuscenes/ \
--log_path ./logs/WaffleIron-48-384__nuscenes/ \
--config ./configs/WaffleIron-48-384__nuscenes.yaml \
--multiprocessing-distributed \
--fp16
```

We used the checkpoint at the *last* training epoch to report the results.

**Remark**: *For single-GPU training, you can remove `--multiprocessing-distributed` and add the argument `--gpu 0`.*


### SemanticKITTI

To retrain the WaffleIron-48-256 backbone, type
```
python launch_train.py \
--dataset semantic_kitti \
--path_dataset /path/to/kitti/ \
--log_path ./logs/WaffleIron-48-256__kitti \
--config ./configs/WaffleIron-48-256__kitti.yaml \
--multiprocessing-distributed \
--fp16
```

At the beginning of the training, the instances for cutmix augmentation are saved in `/tmp/semantic_kitti_instances/`. If this process is interrupted before completion, please delete `/tmp/semantic_kitti_instances/` and relaunch training. You can disable the instance cutmix augmentations by editing the `yaml` config file to set `instance_cutmix` to `False`.

For submission to the official benchmark on the test set of SemanticKITTI, we trained the network on both the val and train sets (argument `--trainval` in `launch_train.py`), used the checkpoint at the last epoch and 12 test time augmentations during inference.


## Creating your own network

### Models

The WaffleIron backbone is defined in `waffleiron/backbone.py` and can be imported in your project by typing
```python
from waffleiron import WaffleIron
```
It needs to be combined with a embedding layer to provide point tokens and a pointwise classification layer, as we do in `waffleiron/segmenter.py`. You can define your own embedding and classification layers instead.


## Preliminary version

To access the preliminary trained models and the corresponding code, you can clone version v0.1.1 of the code.
```
git clone -b v0.1.1 https://github.com/valeoai/WaffleIron
cd WaffleIron/
pip install -e ./
```

The corresponding pretrained models are available at:
```
wget https://github.com/valeoai/WaffleIron/files/10294734/pretrained_nuscenes.tar.gz
tar -xvzf pretrained_nuscenes.tar.gz
wget https://github.com/valeoai/WaffleIron/files/10294735/pretrained_kitti.tar.gz
tar -xvzf pretrained_kitti.tar.gz
```

## Acknowledgements
We thank the authors of 
```
@inproceedings{berman18lovasz,
author = {Berman, Maxim and Triki, Amal Rannen and Blaschko, Matthew B.},
title = {The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure 
in Neural Networks},
booktitle = {CVPR},
year = {2018}
}
```
for making their [implementation](https://github.com/bermanmaxim/LovaszSoftmax) of the Lovász loss publicly available.


## License
WaffleIron is released under the [Apache 2.0 license](./LICENSE). 

The implementation of the Lovász loss in `utils/lovasz.py` is released under 
[MIT Licence](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/LICENSE).
