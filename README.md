## Inverting the Generator of a Generative Adversarial Network

Code for reproducing our experiments in: https://arxiv.org/pdf/1802.05701.pdf

## Data Preparation

1. Download data set

* Download the celebA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

**OR**

* Download the Shoes dataset from [here](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images-square.zip)

**OR**

* Download the Omniglot dataset from [here](https://github.com/brendenlake/omniglot/tree/master/python)

2. Install dependencies listed in req.txt

3. You will also need pyTorch which may be downloaded from [here](http://pytorch.org)

4. Run the relevant notebook for data preprocessing
 
* Run [this](https://github.com/ToniCreswell/attribute-cVAEGAN/blob/master/notebooks/DataToTensorCelebA_smileLabel.ipynb) Jupyter notebook to get the data tensors for CelebA and move them into folder InData/CELEBA/

**OR**

* Run [this](https://github.com/ToniCreswell/InvertingGAN/blob/master/notebooks/shoe2tensor.ipynb) Jupyter notebook to get the data tensors xShoes.npy and yShoes.npy and move them in to folder InData/SHOES/

## Run the code

The code may be run from cmd line with various options detailed in the code

## Example results:

![alt text](https://github.com/ToniCreswell/InvertingGAN/blob/master/imgs/shoes.png)
