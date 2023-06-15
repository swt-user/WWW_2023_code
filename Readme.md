# Implementation for DNS(M,N) and Softmax(\rho, N)
<h2 align="center">
  <img align="center"  src="./fig/relationship_1.png" alt="The Relationship between Hard Negative Sampling and TopK metrics">
</h2>

This is our PyTorch implementation for the paper 2023'WWW:

> Wentao Shi, Jiawei Chen, Fuli Feng, Jizhi Zhang, Junkang Wu, Chongming Gao, Xiangnan He (2023) On the Theories behind Hard Negative Sampling for Recommendation.
[paper link](https://arxiv.org/abs/2302.03472). In WWW 2023.


## Dependencies

- Compatible with PyTorch 1.8.2 and Python 3.8.
- Dependencies can be installed using `requirements.txt`.
## Parameters
+ data
    + gowalla, yelp, amazoni
+ d
    + embedding size
+ m, model
    + 0: `matrix factorization`
    + 1: `NCF`
    + 2: `GMF`
    + 3: `MLP`
    + 4: `LightGCN`
+ sampler
    + 0: `uniform`
    + 2: `AdaSIR uniform`
    + 3: `popularity`
    + 5: `AdaSIR pop`
    
+ loss_type
    + 0: `AdaSIR`
    + 1: `DNS(M, N)`
    + 2: `Softmax(\rho, N)`

## Running Example

`python main_more.py --lambda_w 2 --sampler 0 --sample_num 200 --fix_seed --weighted --loss_type 1` for DNS(M, N)

`python main_more.py --lambda_w 1 --sampler 0 --sample_num 200 --fix_seed --weighted --loss_type 2` for `Softmax(\rho, N)`



## Acknowledgement
The project is built upon [AdaSIR](https://github.com/HERECJ/AdaSIR)


For any clarification, comments, or suggestions please create an issue or contact me.
