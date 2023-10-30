# Simple Example of estimating Heterogeneity using DNNs 
# Max Farrell & Sanjog Misra
# CML 2023


## ----------------------------------------------------------------------------------------------
library(torch)

### generate training data -----------------------------------------------------
# y = alpha(x) + beta(x) t + eps
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
trub = 2-x[, 2, NULL]+.25*(x[, 1, NULL])^3 # This is beta(x)
plot(as.numeric(trub)~as.numeric((x[, 1])),pch=19,col="#00000030")

# True Baseline
# alpha(x) = 0.2 * x_1 - 1.3 * x_2 - 0.5 * x_3
trua= x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5

# Treatment
z <- 0+1*(torch_randn(n, 1L)>0)
table(as.numeric(z))

# Outcomes are linear in treatments
y <- trua+ trub*z+ torch_randn(n, 1)

# Make Binary
y <- 0+1*(y>0)


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

# Loss using torch functions
lossf = nn_bce_loss()

st = proc.time()

for (t in 1:NEPOCH) {
  
  ### -------- Forward pass --------
  ### Causal Model
  # Goal is ATE = E(CATE) = E(b(x))
  
  ab <- model(x) # Output of our deep net (a(x),b(x))
  alpha=ab[,1]$reshape(c(n,1L))
  beta=ab[,2]$reshape(c(n,1L))
  # save the gradient for later use
  ab$retain_grad()
  alpha$retain_grad()
  beta$retain_grad()
  util <- alpha+beta*z # Our model
  # customize the loss to your model
  
  loss <- nnf_binary_cross_entropy_with_logits(util, y, reduction = "mean")
  
  # Other Ideas
  # loss <- nnf_binary_cross_entropy(probs, y, reduction = "mean")
  # How do we get probs?
  # loss = lossf(probs,y)
  
  
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

plot(density(as.numeric(beta)))
lines(density(as.numeric(trub)),lty=2)

# check estimation of beta(x) is meaningful 
mean(trub)
mean(as.numeric(beta))

