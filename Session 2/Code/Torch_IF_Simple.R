# Intro to Torch/AD
# Can we automate derivatives?
# CML 2023
# Max Farrell & Sanjog Misra


library(torch)

# Tensors and AD
z <- torch_tensor(.5, requires_grad = TRUE)
y = z^2+z
y$backward()
z$grad

# Torch IF
set.seed(1234)
n <- 100
x0 <- matrix(runif(n),1,n)
x <- torch_tensor(x0)
w0 = matrix(1,n,1)
w = torch_tensor(w0, requires_grad=TRUE)
mu = torch_matmul(x,w)/torch_sum(w)
grad <- autograd_grad(mu,w)
Lij = ij(data.frame(x=c(x0)),wmean)
Lauto = n*as.numeric(grad[[1]])
plot(Lij~Lauto,pch=19)
abline(0,1)




# Tensors and AD
z <- torch_tensor(.5, requires_grad = TRUE)
y = z^2+z
y$backward()
z$grad