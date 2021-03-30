# smpr3d 
> smpr3d (pronounced 'semper 3D', latin for 'always 3D', short for **S**-**M**atrix **P**hase **R**etrieval & **3D** imaging), simplifies recovering 3D phase-contrast information from scanning diffraction measurements, such as those collected in **4D**-**S**canning **T**ransmission **E**lectron **M**icroscopy (**4D-STEM**) experiments


## Requirements



## Install

`pip install smpr3d`

## Hackathon - How to use on the nesap cluster 

`git clone git@github.com:s-matrix/smpr3d.git`
`module purge`
`module load cgpu`
`module load pytorch/1.8.0-gpu`
`cd smpr3d`
`python setup.py develop --user`
`cd examples`
`sbatch slurm.sh`

## How to contribute
`nbdev_build_lib && nbdev_clean_nbs && nbdev_build_docs`



