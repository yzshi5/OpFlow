# Universal Functional Regression with Neural Operator Flows
The repository contains codes for [Universal Functional Regression with Neural Operator Flows](https://arxiv.org/abs/2404.02986)
![image](https://github.com/yzshi5/OpFlow/assets/109268435/eab9e817-2b81-487c-88fb-90f18f424ed8)




## Requirements
``PyTorch 1.12.1 ``
``scikit-learn 1.2.2 ``


## Files 
| Files | Descriptions|
|-------|-------------|
|**Generation tasks**|
|1D_domain_decomposed_GP.ipynb|placeholder|
|1D_domain_decomposed_TGP.ipynb|placeholder|
|2D_domain_decomposed_GRF.ipynb|placeholder|
|2D_domain_decomposed_TGRF.ipynb|placeholder|
|1D_codomain_GP.ipynb|pass|
|2D_codomain_TGRF.ipynb|pass|
|**Regression tasks**|
|1D_domain_decomposed_GP_prior.ipynb|pass|
|1D_domain_decomposed_GP_Regression.ipynb|pass|
|1D_domain_decomposed_TGP_prior.ipynb|pass|
|1D_domain_decomposed_TGP_Regression.ipynb|pass|
|2D_domain_decomposed_GRF_prior.ipynb|pass|
|2D_domain_decomposed_GRF_Regression_case1.ipynb|pass|
|2D_domain_decomposed_GRF_Regression_case2.ipynb|pass|
|1D_codomain_GP_prior.ipynb|pass|
|1D_codomain_GP_Regression.ipynb|pass|
|**SGLD sampling**|Folder|
|samplers.py|pass|
|SGLD.py|pass|


## Datasets
Synthetic dataset can be directly generated in the training files, earthquake datasets used in the paper can be downloaded from kik-net website [kik-net](https://www.kyoshin.bosai.go.jp/)

## Reference:
@article{shi2024universal,
  title={Universal Functional Regression with Neural Operator Flows},
  author={Shi, Yaozhong and Gao, Angela F and Ross, Zachary E and Azizzadenesheli, Kamyar},
  journal={arXiv preprint arXiv:2404.02986},
  year={2024}
}
