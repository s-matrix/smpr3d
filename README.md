# Welcome to Smpr3D 
> Smpr3d (pronounced 'semper 3D', latin for 'always 3D', short for **S**-**M**atrix **P**hase **R**etrieval & **3D** imaging) simplifies recovering 3D phase-contrast information from scanning diffraction measurements, such as those collected in **4D**-**S**canning **T**ransmission **E**lectron **M**icroscopy (**4D-STEM**) experiments


🚨 **Smpr3D is a research project at an early stage of
development. Expect monstrous bugs and razor-sharp edges!**

![CI](https://github.com/s-matrix/smpr3d/workflows/CI/badge.svg)

## Installing


**TODO** - this project is not on pip or conda yet - install with `python setup.py develop --user`

You can use Smpr3d without any installation by using [Google Colab](https://colab.research.google.com/). In fact, every page of this documentation is also available as an interactive notebook - click "Open in colab" at the top of any page to open it (be sure to change the Colab runtime to "GPU"!).

You can install Smpr3d on your own machines with conda (highly recommended - not working yet :)). If you're using [Anaconda](https://www.anaconda.com/products/individual) then run:
```bash
conda install -c smpr3d -c pytorch -c anaconda smpr3d 
```

To install with pip, use: `pip install smpr3d`. If you install with pip, you should install PyTorch first by following the PyTorch [installation instructions](https://pytorch.org/get-started/locally/).

## Hackathon - How to use on the nesap cluster 

`git clone git@github.com:s-matrix/smpr3d.git`

`module purge`

`module load cgpu`

`module load pytorch/1.8.0-gpu`

`cd smpr3d`

`python setup.py develop --user`

`cd examples`

`sbatch slurm.sh`

## About Smpr3d

A fabulous idea

## Acknowledgements

[Hamish Brown (former NCEM)](https://github.com/HamishGBrown) - theory and first demonstrations

[Colin Ophus (NCEM)](https://github.com/cophus) - theory and first demonstrations

[Alex Rakowski (NCEM)](https://github.com/alex-rakowski) - hpc

[Jim Ciston (NCEM)](https://foundry.lbl.gov/about/staff/jim-ciston/) - first demonstrations

[Mary Scott (NCEM)](https://github.com/orgs/ScottLabUCB/) - first demonstrations

[Scott Findlay (Monash)](https://research.monash.edu/en/persons/scott-findlay) - theory

[Pierre Carrier (HPEnterprise)](https://github.com/PierreCarrier) - performance profiling 

[Daniel Margala (NERSC)](https://github.com/dmargala) - performance profiling

## References

Pelz, P. M. et al. Reconstructing the Scattering Matrix from Scanning Electron Diffraction Measurements Alone. (2020)., `doi <https://arxiv.org/abs/2008.12768v1>`

Brown, H. G. et al. A three-dimensional reconstruction algorithm for scanning transmission electron microscopy data from thick samples. `doi <http://arxiv.org/abs/2011.07652>`

## How to contribute

Before committing, run

`nbdev_build_lib && nbdev_clean_nbs && nbdev_build_docs`

to compile the notebook into script files, clean the notebooks, and build the documentation.



