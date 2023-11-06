# Simple Example estimating Heterogeneity using DNNs
# Max Farrell & Sanjog Misra
# Causal ML 2023
# Basic Idea of Automatic Inference
# ***Important Note: Not to be used in research 
#                    since it doesnt quite follow all steps in FLM. 
# *** For teaching purposes only!!
# We will cover a more "production-ready" version next class 

## ----------------------------------------------------------------------------------------------
library(torch)

### generate training data -----------------------------------------------------
# input dimensionality (number of input features)
d_in <- 5
# output dimensionality (dim of [alpha,beta] : for now 2)
d_out <- 2
# number of observations in training set
n <- 10000

# Treatments
dim_z = 1L

# Set a seed
set.seed(1212)
torch_manual_seed(1212)

## ----------------------------------------------------------------------------------------------
# create random data
x = torch_tensor(matrix(rnorm(n*d_in),n,d_in))

#  True CATE
trub = 2-x[, 2, NULL]+.25*(x[, 1, NULL])^3
plot(as.numeric(trub)~as.numeric((x[, 1])),pch=19,col="#00000030")

# True Baseline
trua= x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5

# Treatment
z <- 0+1*(torch_randn(n, 1L)>0); #torch_randn(n, 1L)*3;

# Outcomes are linear in treatments
y <- trua+ trub*z+ torch_randn(n, 1)

# Logit?
#trupr = 1/(1+exp(-trua-trub*z))
#y=rbinom(n,1,prob=as.numeric(trupr))

# Convert torch tensor
#y = torch_tensor(matrix(y,n,1),dtype = torch_float64())


# The Estimator
## ----------------------------------------------------------------------------------------------
# The architecture (all layers are relu)
arch=c(20,20)

# Model constructor
model <- nn_sequential(nn_linear(d_in, arch[1]),nn_relu())
j=1 # in case single layer
# Loop through architecture
if(length(arch)>1){
  for(j in 2:length(arch)){
    model$add_module(name = paste("layer_",j-1),module=nn_linear(arch[j-1],arch[j]))
    model$add_module(name = paste0("relu_",j-1),nn_relu())
  }
}
# Output layer
model$add_module("ab",nn_linear(arch[j],d_out))


## ----------------------------------------------------------------------------------------------
# Optimization/Learning framework
# for ADAM, need to choose a reasonable learning rate
learning_rate <- 0.01
optimizer <- optim_adam(model$parameters, lr = learning_rate)


## ----------------------------------------------------------------------------------------------
# Training Loop
NEPOCH = 1000
intv = NEPOCH/100
cat("Begining training...\n")
pb <- txtProgressBar(min=1,max=100,style=3)
pct=0
st = proc.time()

for (t in 1:NEPOCH) {

  ### -------- Forward pass --------
  ### Causal Model
  ab <- model(x)
  alpha=ab[,1]$reshape(c(n,1L))
  beta=ab[,2]$reshape(c(n,1L))
  ab$retain_grad()
  y_pred <- alpha+beta*z
  loss <- nnf_mse_loss(y_pred, y, reduction = "mean")

  ### -------- Backpropagation --------
  # Need to zero out the gradients before the backward pass
  optimizer$zero_grad()
  # gradients are computed on the loss tensor
  loss$backward()

  ### -------- Update weights --------
  # use the optimizer to update model parameters
  optimizer$step()
  # progress bar update
  if(t%%intv==0) {pct=pct+1; setTxtProgressBar(pb, pct)}
}
et = proc.time()
elapsed = (et-st)[3]
elapsed

# Now let's do inference
## ----------------------------------------------------------------------------------------------
# Lets extract out parameters
ab <- model(x)
alpha=ab[,1]$reshape(c(n,1L))
beta=ab[,2]$reshape(c(n,1L))

# Inform the graph to retain the grads
ab$retain_grad()
alpha$retain_grad()
beta$retain_grad()

# Plotty Plot plot
plot(as.numeric(trub)~as.numeric((x[, 1])),
     pch=19,col='#00000030',xlab="X",ylab="beta")
points(as.numeric(beta)~as.numeric((x[, 1])),pch=19,col='#ff000030')

# Get predictions
y_pred <- alpha+beta*z

### -------- compute loss --------
loss <- .5*nnf_mse_loss(y_pred, y, reduction = "sum") 
# Why the sum?
# What's with the .5?

# Get Gradients using AD
abg = autograd_grad(loss,ab,create_graph = TRUE,retain_graph = TRUE)

# Dimension
K=dim(abg[[1]])[2]

# Summation Trick!
abg1 = abg[[1]]$sum(1)

# This only works because dab/da = constant
# We'll generalize this next class
hess=matrix(0,K,K)
for(j in 1:K){
tmp=autograd_grad(abg1[j],ab,retain_graph = TRUE)[[1]]
hess[,j] =  as.numeric(tmp$mean(1))
}

# What does this look like?
hess

# Lambda inverse
lami=-solve(hess)

# Gradients
gg = as.array(abg[[1]])

# Repeat to be safe
ab <- model(x)
ab$retain_grad()

# Automatically differentiate H construct
# What is H?
Hi = ab[,2] # What if we wanted something different?
# Sum triakc (again!)
H = Hi$sum()
# Get H_theta
Hab = autograd_grad(H,ab)
Hab2 = (as.array(Hab[[1]]))
# Automatic IF
auto.if = as.numeric(Hi)+sapply(1:n,function(j) Hab2[j,]%*%lami%*%t(gg)[,j])

# Results
au.est= mean(auto.if)
au.se= sqrt(var(auto.if)/n)

# Recall
# Doubly Robust SE
## ----------------------------------------------------------------------------------------------
Z = as.numeric(z)
Y = as.numeric(y)
Yhat = as.numeric(y_pred)
mu0 = as.numeric(alpha)
mu1 = mu0+as.numeric(beta)
e = mean(Z)

IF.adj = ((Z*Y/e - ((Z-e)/e)*mu1)
          - ((1-Z)*Y/(1-e) + ((Z-e)/(1-e))*mu0))
DR.est = mean(IF.adj)
DR.se = sqrt(sum((IF.adj-DR.est)^2)/(n^2))
DR.se

# prettify
dr = c(Est=DR.est,se = DR.se,
       CI.L=DR.est-1.96*DR.se,CI.U=DR.est+1.96*DR.se)
au = c(Est=au.est,se = au.se,
       CI.L=au.est-1.96*au.se,CI.U=au.est+1.96*au.se)

rbind(DRobust=dr,Automatic =au)

