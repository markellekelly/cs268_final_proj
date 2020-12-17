# README

## Source code
`NN.jl` contains the code for a neural network with standard gradient descent, adapted from our source code. `EVGONN.jl` contains very similar code, but is modified to employ the EVGO algorithm.

## Testing and experiments
`run_exp.jl` contains our experiment over values of k. If stored in the same directory as `EVGONN.jl`, it can be run from the command line as-is. Our dataset is downloaded automatically when run, so nothing needs to be downloaded ahead of time.
`compare_gd.ipynb` is a Jupyter notebook that was used to train the final two models, comparing standard and EVGO gradient descent. If stored in the same directory as `NN.jl` and `EVGONN.jl`, it can also be run as-is.