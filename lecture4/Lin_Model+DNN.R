# Code to Understand basic torch implementation of DNNs
# Max Farrell & Sanjog Misra
# CML, 2023

library(torch)

### generate training data -----------------------------------------------------
# input dimensionality (number of input features)
d_in <- 3
# output dimensionality (number of predicted features)
d_out <- 1
# number of observations in training set
n <- 1000


# create random data
x <- torch_randn(n, d_in)
y <- x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5 + torch_randn(n, 1)


### define the network ---------------------------------------------------------

# dimensionality of hidden layer
# d_hidden <- 32

# First a Linear Model
model <- nn_sequential(
  #nn_linear(d_in, d_hidden),
  #nn_relu(),
  #nn_linear(d_hidden, d_out)
  nn_linear(d_in,d_out)
)

### network parameters ---------------------------------------------------------

# for adam, need to choose a much higher learning rate in this problem
learning_rate <- 0.01

optimizer <- optim_adam(model$parameters, lr = learning_rate)

### training loop --------------------------------------------------------------

for (t in 1:1000) {
  
  ### -------- Forward pass -------- 
  
  y_pred <- model(x)
  
  ### -------- compute loss -------- 
  loss <- nnf_mse_loss(y_pred, y, reduction = "sum")
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  ### -------- Backpropagation -------- 
  
  # Need to zero out the gradients before the backward pass, only this time,
  # on the optimizer object
  optimizer$zero_grad()
  
  # gradients are still computed on the loss tensor (no change here)
  loss$backward()
  
  ### -------- Update weights -------- 
  
  # use the optimizer to update model parameters
  optimizer$step()
}


theta = c(as.matrix(model$parameters$`0.bias`),as.matrix(model$parameters$`0.weight`))
cbind(coef(lm(as.matrix(y)~as.matrix(x))),theta)

