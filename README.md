# Flowification

This repository contains the implementation of the paper

> [Flowification: Everything is a Normalizing Flow](https://arxiv.org/abs/2205.15209) by Bálint Máté, Samuel Klein, Tobias Golling and François Fleuret.


### Environment
The environment that is necessary to run our code can be created by
````
conda create --name flowification python=3.8
conda activate flowification
conda install pytorch==1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install wandb nflows
````
### Experiments
From within the above environment the following command can then be executed

```
python3 experiments/image_generation.py --data DATASET --architecture MODEL
```
where DATASET should be either `mnist` or `cifar` and MODEL should be a member of `[mlp, conv1, conv2, conv1_nsf, conv2_nsf]`.

### Logging to weights and biases
By default all logging happens offline to the `wandb`directory. If you would like the results to be pushed to your wandb account, put your wandb key in the `wandb.key` file at the root of the repo.


## BibTeX

If you find this repository or our paper useful, consider citing us at

```
@article{2022flowification,
  title={Flowification: Everything is a Normalizing Flow},
  author={M{\'a}t{\'e}, B{\'a}lint and Klein, Samuel and Golling, Tobias and Fleuret, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:2205.15209},
  year={2022}
}
```
