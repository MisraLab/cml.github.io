# Automatic Inference on Causal Deepnets
# CML 2023
# Instructors: Max Farrell & Sanjog Misra
# Code to do automatic inference on statistics
# based on first stage DNN parameter functions
# In this example we will focus on the ATE = E(CATE)
# The example below uses a linear model but can be extended to 
# other loss functions (as discussed in class) easily.
# Note: The code is for teaching purposes only
# It should serve as a guide to building your own applciations
# The code is provided "as-is" with no guarantees of completeness or 
# accuracy. The authors assume no responsibility or liability for any errors or omissions.
# Please consider citing this paper if you find the contents useful:
# https://arxiv.org/abs/2010.14694
# Last Edited: Nov 19, 2023 by Sanjog Misra


## Load Torch
library(torch)
library(MASS)

# Same old Same old - 
### generate training data -----------------------------------------------------
# Model is:  y = alpha(x) + beta(x) t + eps

# input dimensionality (number of input features 'X')
d_in <- 5
# output dimensionality (dim of [alpha,beta] : for now 2)
# Recall the output is the number of parameter functions
# Code allows for more that 2...
d_out <- 2

# number of observations in training set
n <- 15000

# Treatments (connected to d_out)
dim_z = 1L

# Set a seed
set.seed(1212)
torch_manual_seed(1212)

## ----------------------------------------------------------------------------------------------
# create random data
x = torch_tensor(matrix(rnorm(n*d_in),n,d_in))

#  True CATE
#  beta(x) = 2 - x_2 + 0.25 * x_1^3
trub = 2-x[, 2, NULL]+.25*(x[, 1, NULL])^3 # This is beta(x)
plot(as.numeric(trub)~as.numeric((x[, 1])),pch=19,col="#00000030")

# True Baseline
# alpha(x) = 0.2 * x_1 - 1.3 * x_2 - 0.5 * x_3
trua= x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5

# Treatment (constant propensity)
z <- 0+1*(torch_randn(n, 1L)>0)

# Treatment (non-constant propensity)
# Uncomment next 4 lines to test

trup = 1/(1+exp(1+.5*x[,2]))
z = rbinom(n,1,prob=as.numeric(trup))
Back to tensor
z = torch_tensor(as.matrix(z))

table(as.numeric(z))

# Outcomes are linear in treatments
y <- trua+ trub*z+ torch_randn(n, 1)

# Check something
summary(lm(as.numeric(y)~as.numeric(z)))

# Full dataset
full_dat = list()
full_dat$x= x
full_dat$y= y
full_dat$z= z

# sample split into 3 splits
idxc= idx=list()
idx[[1]] = sample(1:n,size = n/3,replace=FALSE)
idxc[[1]] = setdiff(1:n,idx[[1]])

idx[[2]] = sample(idxc[[1]],size = n/3,replace=FALSE)
idxc[[2]] = setdiff(1:n,idx[[2]])

idx[[3]] = setdiff(idxc[[1]],idx[[2]])
idxc[[3]] = setdiff(1:n,idx[[3]])

# Create three datasets 
  d1 = list()
  d1$x=x[idx[[1]],]
  d1$y=y[idx[[1]],]
  d1$z=z[idx[[1]],]

# Dataset split 2 
  d2 = list()
  d2$x=x[idx[[2]],]
  d2$y=y[idx[[2]],]
  d2$z=z[idx[[2]],]
  
# Dataset split 3 
  d3 = list()
  d3$x=x[idx[[3]],]
  d3$y=y[idx[[3]],]
  d3$z=z[idx[[3]],]

# Code to implement causalDNN 
  source('cdnn.R') # Step 1
  source('linDNN.R') ; source('projLam.R') # Step 2
  source('procRes.R') # Step 3 
    
# Ok Let's do this!
# We will use the following cross-fits
# DNN, E(L|X), Statistic (averaging)   
# 1,2,3
# 3,1,2
# 2,3,1  
# Why these not others?  

# Run DeepNets      
dnn1 = cdnn(d1,arch=c(20,20))
dnn3 = cdnn(d3,arch=c(20,20))
dnn2 = cdnn(d2,arch=c(20,20))

# Get parameters on full data 
aba = dnn1$model(x)
abb = dnn2$model(x)
abc = dnn3$model(x)

# Plots
plot(as.numeric(trub)~as.numeric((x[, 1])),pch=19,col="#00000030")
points(as.numeric(aba[,2])~as.numeric(x[,1]),pch=19,col='#ff000020')
points(as.numeric(abb[,2])~as.numeric(x[,1]),pch=19,col='#f000ff20')
points(as.numeric(abc[,2])~as.numeric(x[,1]),pch=19,col='#00A00020')

# Density Plots Check
plot(density(as.numeric(trub)))
lines(density(as.numeric(as.numeric(aba[,2]))),col='red')
lines(density(as.numeric(as.numeric(abb[,2]))),col='blue')
lines(density(as.numeric(as.numeric(abc[,2]))),col='green')

# Projections
# Projects the Hessian onto X
# To form conditional expectation functions
lProj2 = makeLam(dat = d2,dnn=dnn1)
lProj1 = makeLam(dat = d1,dnn=dnn3) 
lProj3 = makeLam(dat = d3,dnn=dnn2) 

# Compute IF for each split and stack
# What statistic are we interested in
# Let's say ATE = (E[H]=E(CATE)=E(ab[,2]))
H=function(ab,dat) ab[,2]

# Use split 3s for 
fin3 = proc_res(d3,dnn1,lProj2,H)
fin2 = proc_res(d2,dnn3,lProj1,H)
fin1 = proc_res(d1,dnn2,lProj3,H)

# Stack 
af1 = fin1$auto.if
af2 = fin2$auto.if
af3 = fin3$auto.if
cf.est= mean(c(af1,af2,af3))
cf.se = sqrt((1/3)*(var(af1)+var(af2)+var(af3))/n)

cf = c(Est=cf.est,se = cf.se,
       CI.L=cf.est-1.96*cf.se,CI.U=cf.est+1.96*cf.se)

cf

#Truth
mean(trub)


