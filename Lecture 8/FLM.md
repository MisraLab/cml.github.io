Unzip files and run main_cdnn_CrossFit.R. 
The code walks through a simple example:

1. Estimates DNN for {a(x),b(x)} in Y=a(x)+b(x)T+err model.
2. Projects Hessian of loss on to covaraites (x) using DNN
3. Compute se/CI for ATE. mu=E(b(x))

The code can handle multiple treatments, non-constant propensity scores and can be adapted to arbitrary loss functions. 

Usual Disclaimer: The code is provided "as-is" with no guarantees of completeness or accuracy. The authors assume no responsibility or liability for any errors or omissions.

Please consider citing this paper if you find the contents useful:
https://arxiv.org/abs/2010.14694