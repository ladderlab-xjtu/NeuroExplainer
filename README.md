# NeuroExplainer: Fine-Grained Attention Decoding to Uncover Cortical Development Patterns of Preterm Infants

## Description
NeuroExplainer is an explainable geometric deep network adopting a hierarchical attention-decoding framework to learn fine-grained attention and discriminative representations in a spherical space to accurately recognize preterm infants from term-born infants.
NeuroExplainer learns the hierarchical attention-decoding modules under subject-level weak supervision coupled with targeted regularizers deduced from domain knowledge regarding brain development. These prior-guided constraints implicitly maximize the explainability metrics (i.e., fidelity, sparsity, and stability) in network training, driving the learned network to output detailed explanations and accurate classifications.

![](https://github.com/qianyuhou/NeuroExplainer/blob/main/images/architecture.png)

![](https://github.com/qianyuhou/NeuroExplainer/blob/main/images/attention-comparison.png)
## Package Dependency
- python (3.6)
- pytorch (0.4.1+)
- torchvision (0.2.1+)
- tensorboardx (1.6+)
- NumPy (1.11.3)
- SciPy (1.2.1)
- pyvista (0.22.4+)
## Step 0. Environment setup
```
git clone https://github.com/ladderlab-xjtu/NeuroExplainer.git
```
You can use conda to easily create an environment for the experiment using following command:
```
conda create -n neuroexplainer python=3.6 
conda activate neuroexplainer
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c conda-forge pyvista
```
To install the required packages, run:
```
pip install .
```

The public dataset was used in this work (thank the authors for sharing their datasets!):
- [dHCP]
## Step 1. Data preparation
## Step 2. Training
### Train
### Test
### Visualization
## Citation
Please cite the following paper if you use (part of) our code in your research:
```
@article{Ha2022:SPHARMNet,
  author    = {Ha, Seungbo and Lyu, Ilwoo},
  journal   = {IEEE Transactions on Medical Imaging},
  title     = {SPHARM-Net: Spherical Harmonics-Based Convolution for Cortical Parcellation},
  year      = {2022},
  volume    = {41},
  number    = {10},
  pages     = {2739-2751},
  doi       = {10.1109/TMI.2022.3168670},
  publisher = {IEEE}
}
```
