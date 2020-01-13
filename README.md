# Spatial GAN 

## The network

The network presented here is a straight-up implementation of the spatial GAN in PyTorch (ported over from the author's Lasgane implementation https://github.com/zalandoresearch/spatial_gan ) as well as an adaptation to use the Wasserstein method for training.

The spatial GAN is described in this paper: https://arxiv.org/abs/1611.08207

Wasserstein GAN (WGAN) training is described in this paper: https://arxiv.org/abs/1701.07875


The WGAN is currently not working as well as expected with my datasets, so it is work in progress.

### Requirements

You will need to install PIL `pip install Pillow` and PyTorch.

### Usage 

I do not provide the dataset so first you need to create a dataset:
 - make a directory called dataset
 - put PNG files of your pictures in there
 - create a file called dataset_labels.csv where the first line is irrelevant then each line is the name of one of the picture without the png extension, something like

````
filenames
flower1
flower2
flowerbig
````

It indicates which files to use for training. In this case it corresponds to the files flower1.png, flower2.png, flowerbig.png.

If this is inconvenient, feel free to modify LoadData to load your dataset.

Important note: the network takes input in the range [-1, 1] so make sure you normalize your data.

### Screenshots

Here is a screenshot of a successful run
![Flowers](https://raw.githubusercontent.com/edeguine/wsgan/master/samples/stored_floralbig_gen_3300_1.png)


I provide a trained model, you can use it as such:

`python generate.py models/gen_3300.pth`


## The code

### License

The software is free but copyrighted. It is copyrighted under the [JRL license](https://en.wikipedia.org/wiki/Java_Research_License), commercial or proprietary use is forbidden but research and academic use are allowed.

### Implementation details

#### Vanilla implementation

The vanilla implementation follows the paper. An important detail is that the batchnorm happens before the activation function.

#### Wasserstein WGAN implementation

The WGAN implementation follows reference implementation for WGANs. It adds a Dense layer to the discriminator to turn it into a 'critic'.
