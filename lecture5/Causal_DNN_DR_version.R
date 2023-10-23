# Simple Example of estimating Heterogeneity using DNNs 
# Max Farrell & Sanjog Misra
# CML 2023


## ----------------------------------------------------------------------------------------------
library(torch)

### generate training data -----------------------------------------------------
# y = alpha(x) + beta(x)t + eps
# input dimensionality (number of input features)
d_in <- 5
# output dimensionality (dim of [alpha,beta] : for now 2)
d_out <- 2
# number of observations in training set
n <- 20000

# Treatments
dim_z = 1L

# Set a seed
set.seed(1212)
torch_manual_seed(1212)

## ----------------------------------------------------------------------------------------------
# create random data
x = torch_tensor(matrix(rnorm(n*d_in),n,d_in))

#  True CATE
#  beta(x) = 2 - x_2 + 0.25 * x_1^3
trub = 2-x[, 2, NULL]+.25*(x[, 1, NULL])^3
plot(as.numeric(trub)~as.numeric((x[, 1])),pch=19,col="#00000030")

# True Baseline
# alpha(x) = 0.2 * x_1 - 1.3 * x_2 - 0.5 * x_3
trua= x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5

# Treatment
z <- 0+1*(torch_randn(n, 1L)>0)
table(as.numeric(z))

# Outcomes are linear in treatments
y <- trua+ trub*z+ torch_randn(n, 1)

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

# see the architecture
model

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
  # save the gradient for later use
  ab$retain_grad()
  alpha$retain_grad()
  beta$retain_grad()
  y_pred <- alpha+beta*z
  # customize the loss to your model
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
ab <- model(x)
alpha=ab[,1]$reshape(c(n,1L))
beta=ab[,2]$reshape(c(n,1L))
ab$retain_grad()
alpha$retain_grad()
beta$retain_grad()

plot(as.numeric(trub)~as.numeric((x[, 1])),
     pch=19,col='#00000030',xlab="X",ylab="beta")
points(as.numeric(beta)~as.numeric((x[, 1])),pch=19,col='#ff000030')

# check estimation of beta(x) is meaningful 
mean(trub)
mean(as.numeric(beta))

# Inference
# Doubly Robust SE
## ----------------------------------------------------------------------------------------------
Z = as.numeric(z)
Y = as.numeric(y)

Yhat = as.numeric(y_pred)
mu0 = as.numeric(alpha)
mu1 = mu0+as.numeric(beta)
e = mean(Z)

# As Max showed in class
# clip when est. propensity score too close to 0 or 1
IF = (mu1 + Z*(Y-mu1)/e) - (mu0+ (1-Z)*(Y-mu0)/(1-e))
DR.est = mean(IF)
DR.se = sqrt(var(IF)/n)



# Stack and check answers from the two approaches
dr = c(Est=DR.est,se = DR.se,
       CI.L=DR.est-1.96*DR.se,CI.U=DR.est+1.96*DR.se)

print(dr)


# Some other approaches for comparison
# LASSO
library(glmnet)

# Transform tensors back to R matrices
xd = as.matrix(x)
yd = as.matrix(y)
zd = as.matrix(z)

# Lasso (with Cross validation)
g0 = cv.glmnet(y=yd[zd==0,],x=xd[zd==0,])
g1 = cv.glmnet(y=yd[zd==1,],x=xd[zd==1,])

# Same as before
mu0 = predict(g0,newx=xd)
mu1 = predict(g1,newx=xd)

# CATE and ATE
tauhat = mu1-mu0
mean(tauhat)

# Use the Lasso estimates
IF.lasso = (mu1 + Z*(Y-mu1)/e) - (mu0+ (1-Z)*(Y-mu0)/(1-e))
DRL.est = mean(IF.lasso)
DRL.se = sqrt(var(IF.lasso)/n)

# Stack and check answers from the two approaches
drL = c(Est=DRL.est,se = DRL.se,
        CI.L=DRL.est-1.96*DRL.se,CI.U=DRL.est+1.96*DRL.se)

rbind(dr,drL)

# How do we do on CATEs?
plot(as.numeric(trub)~as.numeric((x[, 1])),
     pch=19,col='#00000030',xlab="X",ylab="beta")
points(as.numeric(beta)~as.numeric((x[, 1])),pch=19,col='#ff000030')
points(as.numeric(tauhat)~as.numeric((x[, 1])),pch=19,col='#ff00ff30')


# Let's try Causal Forests
library(grf)
cf0 = causal_forest(X=xd,Y=yd,W=zd)
cf.tau = predict(cf0)$pred
res.cf = average_treatment_effect(cf0)

# Stack and check answers from the two approaches
drCF = c(Est=res.cf[1],se = res.cf[2],
         CI.L=res.cf[1]-1.96*res.cf[2],CI.U=res.cf[1]+1.96*res.cf[2])

rbind(dr,drL,drCF)


# Add to plot
points(as.numeric(cf.tau)~as.numeric((x[, 1])),pch=19,col='#ffff0030')




