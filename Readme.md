# Implementation for AdaSIR
"Learning Recommenders for Implicit Feedback with Importance Resampling" (WWW2022)
The codes are tested in Pytorch

## Parameters
+ data
    + gowalla, yelp, amazoni
+ d
    + embedding size
+ m, model
    + 0: matrix factorization
    + 1: NCF
    + 2: GMF
    + 3: MLP
+ sampler
    + 0: uniform
    + 2: AdaSIR uniform
    + 3: popularity
    + 5: AdaSIR pop
    + 7: AdaSIR uniform + rank estimation
    + 8: DNS
    + 9: Adaptive kernel(only works for matrix factorization)

## Running Example

`python main_more.py --sampler 0 --weighted` for PRIS(U)

`python main_more.py --sampler 2 --weighted` for AdaSIR

or

`sh run.sh`
