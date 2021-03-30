# Welcome to smpr3d 
> smpr3d (pronounced 'semper 3D', latin for 'always 3D', short for **S**-**M**atrix **P**hase **R**etrieval & **3D** imaging), simplifies recovering 3D phase-contrast information from scanning diffraction measurements, such as those collected in **4D**-**S**canning **T**ransmission **E**lectron **M**icroscopy (**4D-STEM**) experiments


## Requirements



## Installing

TODO - check this works

You can use smpr3d without any installation by using [Google Colab](https://colab.research.google.com/). In fact, every page of this documentation is also available as an interactive notebook - click "Open in colab" at the top of any page to open it (be sure to change the Colab runtime to "GPU" to have it run fast!) See the fast.ai documentation on [Using Colab](https://course.fast.ai/start_colab) for more information.

You can install smpr3d on your own machines with conda (highly recommended). If you're using [Anaconda](https://www.anaconda.com/products/individual) then run:
```bash
conda install -c smpr3d -c pytorch -c anaconda smpr3d 
```

To install with pip, use: `pip install fastai`. If you install with pip, you should install PyTorch first by following the PyTorch [installation instructions](https://pytorch.org/get-started/locally/).

## Hackathon - How to use on the nesap cluster 

`git clone git@github.com:s-matrix/smpr3d.git`

`module purge`

`module load cgpu`

`module load pytorch/1.8.0-gpu`

`cd smpr3d`

`python setup.py develop --user`

`cd examples`

`sbatch slurm.sh`

## About smpr3d



## How to contribute
`nbdev_build_lib && nbdev_clean_nbs && nbdev_build_docs`



