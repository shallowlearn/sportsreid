<img src="https://github.com/sportsreid/sportsreid/blob/main/images/hierarchical_8921.jpg" width="450" height=240>  &emsp;&emsp;&emsp;&emsp;&emsp;        <img src="https://github.com/sportsreid/sportsreid/blob/main/images/hierarchical_3297.jpg" width="450" height=240>
# Sportsreid

Sportsreid is useful for re-identifying the same player in different frames of a broadcast video of a match. This repo is built on top of [SoccerNet Re-Identification](https://github.com/SoccerNet/sn-reid) and the popular [Torchreid](https://github.com/KaiyangZhou/deep-person-reid). It is currently ranked #2 on the test split leaderboard for the SoccerNet 2022 ReIdentification challenge.

**Note to reviewers: At the time of submission of our manuscript, our approach was #1 on the test split leaderboard. Also note that our models are pretrained on only Imagenet. We do not have any such details about the approach that is currently #1 on the leaderboard.**

## Metrics and Pretrained Models

| name | #params | Resolution | mAP | rank-1 | chkpt | config |
| ---  | --- | --- | --- | --- | --- | --- |
| ResNet50-fc512 | 22.0 | 256x128 | 81.8 | 76.1 | [model](https://drive.google.com/file/d/1o45E8lxB9mxJ1lfSgMpi3mC0zUwVvzgz/view?usp=sharing) | [config](https://drive.google.com/file/d/1CqtCPpn9NSlZ5NMmGUqWfd-fcOOWVyOu/view?usp=sharing) |
| OSNet_x1_0 | 22.0 | 256x128 | 83.4 | 78.0 | [model](https://drive.google.com/file/d/1To0Ww6_HxU2ITAlb4kQEgYExV-orwit8/view?usp=sharing) | [config](https://drive.google.com/file/d/1xO4Qe7f4FwpXnEe39cn24FdRDg6F-LLu/view?usp=sharing) |
| DeiT-Tiny/16 | 86.6 | 224x224 | 82.2 | 76.2 | [model](https://drive.google.com/file/d/1u6SLzk8TTZQt2NvNrE2KNOFB6S67JE1g/view?usp=sharing) | [config](https://drive.google.com/file/d/1wZCXISaGdeyQfgL1MaN7BNVKOZTBwku-/view?usp=sharing) |
| DeiT-S/16 | 86.9 | 224x224 | 84.3 | 79.4 | [model](https://drive.google.com/file/d/1yPgYoxP5a8X5p0LGdB_YHAXHDyRcShIo/view?usp=sharing) | [config](https://drive.google.com/file/d/1wZCXISaGdeyQfgL1MaN7BNVKOZTBwku-/view?usp=sharing) |
| ViT-B/16 | 304.4 | 224x224 | 86.0 | 81.5 | [model](https://drive.google.com/file/d/1yoGabayh4yRGkfzBwmBF1ocNiwwo1WVv/view?usp=sharing) | [config](https://drive.google.com/file/d/1f7JL1sBqThM9J3lL4XrTEAA2k3bVAOl3/view?usp=sharing) |
| ViT-L/16* | 304.8 | 224x224 | 89.8 | 86.7 | [model](https://drive.google.com/file/d/1NHtpTuCCueA1Q8S5li3y4L3oO9lFvte2/view?usp=sharing) | [config](https://drive.google.com/file/d/1ZdHOJwben0drx2xJcaf-UczgKzKReWY6/view?usp=sharing) |

The ViT-L/16* model is trained with 5 different random seeds for initialization and then the weights are averaged across these seeds to further increase mAP. This is inspired by this [paper](https://arxiv.org/abs/2203.05482).

## Installation

```
# cd to your preferred directory and clone this repo
git clone https://github.com/sportsreid/sportsreid.git

# create environment
cd sportsreid/
conda create --name sportsreid python=3.7
conda activate sportsreid

# install dependencies
# make sure `which python` and `which pip` point to the correct path
pip install -r requirements.txt

# install torch and torchvision (select the proper cuda version to suit your machine)
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# install sportsreid (don't need to re-build it if you modify the source code)
python setup.py develop
```




