# NeuroExplainer: Fine-Grained Attention Decoding to Uncover Cortical Development Patterns of Preterm Infants

## Description
NeuroExplainer is an explainable geometric deep network adopting a hierarchical attention-decoding framework to learn fine-grained attention and discriminative representations in a spherical space to accurately recognize preterm infants from term-born infants.
NeuroExplainer learns the hierarchical attention-decoding modules under subject-level weak supervision coupled with targeted regularizers deduced from domain knowledge regarding brain development. These prior-guided constraints implicitly maximize the explainability metrics (i.e., fidelity, sparsity, and stability) in network training, driving the learned network to output detailed explanations and accurate classifications.

### The schematic diagram of our NeuroExplainer architecture and Spherical attention mechanism:
![](https://github.com/qianyuhou/NeuroExplainer/blob/main/images/architecture.png)
### Typical examples of the explanation factors captured by different methods:
![](https://github.com/qianyuhou/NeuroExplainer/blob/main/images/attention-comparison.png)

## Package Dependency
- python (3.10)
- pytorch (2.1.0)
- torchvision (1.16.0)
- tensorboardx (2.6.2.2)
- NumPy (1.22.4)
- SciPy (1.11.3)
- pyvista (0.42.3)

## Step 0. Environment setup
```
git clone https://github.com/ladderlab-xjtu/NeuroExplainer.git
```
You can use conda to easily create an environment for the experiment using following command:
```
conda create -n neuroexplainer python=3.10
conda activate neuroexplainer
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyvista
```
To install the required packages, run:
```
pip install .
```

## Step 1. Data preparation
We provide processed and curated dataset from the [Developing Human Connectome Project (dHCP)](https://biomedia.github.io/dHCP-release-notes/index.html). Note that the data repository is private and only visible to those who have access. The input spherical surfaces contain 10, 242 vertices, and three features, i.e., cortical thickness, mean curvature, and convexity, are required for the classfication.
## Step 2. Training

After data prepration, modify the [main.py](https://github.com/qianyuhou/NeuroExplainer/blob/main/main.py) file to match the training data in your own path. To save the coresponding models and results, the save directories must be manually changed. Then, run:
```
python main.py
```
If successful, you will see
```
6952268 paramerters in total
learning rate = 
[0:0/247]  LOSS=  LOSS_1=  LOSS_2=  LOSS_3=  LOSS_4=  LOSS_5=
[0:1/247]  LOSS=  LOSS_1=  LOSS_2=  LOSS_3=  LOSS_4=  LOSS_5=
Train_arr=
Val_acc=   ,Val_auc=
save max acc or auc model
Val_max_auc=   ,Val_max_arr=   ,save_epoch=   ,save_num=
last five train Dice: 
All complete in ...
```

In our implementation, the feature representations produced by EB-1 to EB-4 in [architecture](https://github.com/qianyuhou/NeuroExplainer/blob/main/images/architecture.png) have 32, 64, 128, and 256 channels, respectively. Correspondingly, DB-1 to DB-3, and the final classification layer have 256, 128, 64, and 32 channels, respectively.

The GPU memory consumption may vary depending on CUDA kernels.
### Step 3. Test
In this step, you can calculate the accuracy of classification for preterm and term-born infants and obtain the output attention maps on surface and sphere. To predict a single surface’ attention map, you need to modify the [test.py](https://github.com/qianyuhou/NeuroExplainer/blob/main/test.py) file to match the data in your own path. To save the attention maps of the left and right brain at the sphere and surface, the save directories must be manually changed or created. In this step, you can use the saved model `./muilt_view_10242_ori_$epoch_max_acc.pkl` in Step 2.
### Visualization
You can use [Paraview](https://www.paraview.org/) software to visualize the attention map in VTK format. An example of the coarse-grained attention map and the fine-grained attention map of preterm infant are shown below. More usages about Paraview please refer to [Paraview](https://www.paraview.org/).
![paraview](https://github.com/qianyuhou/NeuroExplainer/blob/main/images/attention%20map.png)
## Citation
Please cite the following paper if you use (part of) our code in your research:
```
Xue, C., Wang, F., Zhu, Y., Li, H., Meng, D., Shen, D., & Lian, C.(2023). NeuroExplainer: Fine-Grained Attention Decoding to Uncover Cortical Development Patterns of Preterm Infants. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14221. Springer, Cham.
```
