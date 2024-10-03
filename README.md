# Universal Functional Regression with Neural Operator Flows
The repository contains codes for [Universal Functional Regression with Neural Operator Flows](https://arxiv.org/abs/2404.02986)

(Appeared at Transaction on Machine Learning Research (TMLR), 2024 by Shi, Yaozhong and Gao, Angela F and Ross, Zachary E and Azizzadenesheli, Kamyar)

![image](https://github.com/yzshi5/OpFlow/assets/109268435/eab9e817-2b81-487c-88fb-90f18f424ed8) 
 



## Requirements
``PyTorch 1.12.1 ``
``scikit-learn 1.2.2 ``


## Files 
| Files | Descriptions|
|-------|-------------|
|**Generation tasks**|
|1D_domain_decomposed_GP.ipynb|resolution=256, generation task for 1D GP data|
|1D_domain_decomposed_TGP.ipynb|resolution=256, generation task for 1D Truncated GP data|
|2D_domain_decomposed_GRF.ipynb|resolution=64x64, generation task for 2D GRF data|
|2D_domain_decomposed_TGRF.ipynb|resolution=64x64, generation task for 2D Truncated GRF data|
|1D_codomain_GP.ipynb|resolution=256, **sliding regularization** used, generation task for 1D GP data, codomain OpFlow|
|2D_codomain_TGRF.ipynb|resolution=64x64, generatin tasks for 2D Truncated GRF data, codomain OpFlow|
|**Regression tasks**|
|1D_domain_decomposed_GP_prior.ipynb|resolution=128|
|1D_domain_decomposed_GP_regression.ipynb|duplicate the results of classical GPR|
|1D_domain_decomposed_TGP_prior.ipynb|resolution=128|
|1D_domain_decomposed_TGP_regression.ipynb|Non-Gaussian process regression|
|2D_domain_decomposed_GRF_prior.ipynb|resolution=32x32|
|2D_domain_decomposed_GRF_regression_case1.ipynb|regression with scatter observations|
|2D_domain_decomposed_GRF_regression_case2.ipynb|regression with strip observations|
|1D_codomain_GP_prior.ipynb|resolution=128|
|1D_codomain_GP_regression.ipynb|codomain GP Regression|
|**SGLD sampling**|
|samplers.py|
|SGLD.py|
|**Comments**| **sliding regularization** trick used in some files can be useful for others challenging tasks, feel free to add that on for all tasks|

## Datasets
Synthetic dataset can be directly generated in the training files, earthquake datasets used in the paper can be downloaded from kik-net website [kik-net](https://www.kyoshin.bosai.go.jp/)

## Reference:
@article{shi2024universal,
  title={Universal Functional Regression with Neural Operator Flows},
  author={Shi, Yaozhong and Gao, Angela F and Ross, Zachary E and Azizzadenesheli, Kamyar},
  journal={arXiv preprint arXiv:2404.02986},
  year={2024}
}
