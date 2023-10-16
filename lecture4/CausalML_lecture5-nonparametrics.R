##########################################################################
##	Class: BUS 41917 Causal Machine Learning
##	Lecture 5 code
##	Max H. Farrell & Sanjog Misra
##  Winter 2023
##
##	Purpose:
##		-Demonstrate nonparametric regression using binning and deep nets
##
##	Data files:
##		-none required
##
##
##########################################################################


rm(list=ls())

library(torch)
library(binsreg)
library(rpart)
library(randomForest)



### Function Approximation ###


# ggplot is annoying, so I wrote a stupid function to quickly get what we need for class
binsregplot <- function(bins=2,deg=0, fcn=TRUE) {
  bs <- binsreg(y.R,x.R, dots=c(deg,0), binspos="es", nbins=bins, line=c(deg,0))
  a <- bs$data.plot$`Group Full Sample`$data.line
  plot(x.R,y.R, type="l",col="blue",lwd=2, lty="dashed")
  lines(sort(x.R),0.5 + sin(10*x.R)[order(x.R)],col="blue",lwd=2, lty="dashed")
  lines(a$x,a$fit,col="red",lwd=2)
}


x.R <- seq(0,1,length=1000)
y.R <- 0.5 + sin(10*x.R)
plot(x.R,y.R, type="l",col="blue",lwd=2, lty="dashed")


binsregplot(bins=1,deg=0)
binsregplot(bins=1,deg=1)
binsregplot(bins=1,deg=2)
binsregplot(bins=1,deg=3)
binsregplot(bins=1,deg=4)
binsregplot(bins=1,deg=5)


binsregplot(bins=1,deg=0)
binsregplot(bins=2,deg=0)
binsregplot(bins=4,deg=0)
binsregplot(bins=8,deg=0)
binsregplot(bins=16,deg=0)

binsregplot(bins=1,deg=1)
binsregplot(bins=2,deg=1)
binsregplot(bins=4,deg=1)
binsregplot(bins=8,deg=1)
binsregplot(bins=16,deg=1)

binsregplot(bins=1,deg=2)
binsregplot(bins=2,deg=2)
binsregplot(bins=4,deg=2)
binsregplot(bins=8,deg=2)





### Now with actual data ###

# Set up DGP and draw a random sample

# number of observations in training set
  n <- 100

# create random data
x <- torch_rand(n,1)
e <- torch_randn(n,1)/3
y <- 0.5 + sin(10*x[,1,NULL]) + torch_randn(n,1)/3

y.R <- as.numeric(y)
x.R <- as.numeric(x)

plot(x,y, ylim = c(-1.5,2.5))
plot(x.R,y.R, ylim = c(-1.5,2.5))
lines(sort(x.R),0.5 + sin(10*x.R)[order(x.R)],col="blue",lwd=2, lty="dashed")

## ggplot is annoying, so I wrote a stupid function to quickly get what we need for class
binsregplot <- function(bins=2,deg=0) {
  bs <- binsreg(y.R,x.R, dots=c(deg,0), binspos="es", nbins=bins, line=c(deg,0), masspoints="off")
  a <- bs$data.plot$`Group Full Sample`$data.line
  plot(x.R,y.R, ylim = c(-1.5,2.5))
  lines(sort(x.R),0.5 + sin(10*x.R)[order(x.R)],col="blue",lwd=2, lty="dashed")
  lines(a$x,a$fit,col="red",lwd=2)
}


binsregplot(bins=1,deg=0)
binsregplot(bins=1,deg=1)
binsregplot(bins=1,deg=2)
binsregplot(bins=1,deg=3)
binsregplot(bins=1,deg=4)
binsregplot(bins=1,deg=5)

binsregplot(bins=1,deg=0)
binsregplot(bins=2,deg=0)
binsregplot(bins=4,deg=0)
binsregplot(bins=8,deg=0)
binsregplot(bins=16,deg=0)
#optimal number of bins:
binsregselect(y.R,x.R, bins=c(0,0), binspos="es", masspoints="off")


binsregplot(bins=1,deg=1)
binsregplot(bins=2,deg=1)
binsregplot(bins=4,deg=1)
binsregplot(bins=8,deg=1)
binsregplot(bins=16,deg=1)
#optimal number of bins:
binsregselect(y.R,x.R, bins=c(1,0), binspos="es", masspoints="off")

binsregplot(bins=1,deg=2)
binsregplot(bins=2,deg=2)
binsregplot(bins=4,deg=2)
binsregplot(bins=8,deg=2)
#optimal number of bins:
binsregselect(y.R,x.R, bins=c(2,0), binspos="es", masspoints="off")


tree <- rpart(y.R ~ x.R)
plot(x.R,y.R)
lines(sort(x.R),0.5 + sin(10*x.R)[order(x.R)],col="blue",lwd=2, lty="dashed")
lines(sort(x.R),predict(tree)[order(x.R)], col="red", lwd=2)

forest <- randomForest(y.R ~ x.R)
plot(x.R,y.R)
lines(sort(x.R),0.5 + sin(10*x.R)[order(x.R)],col="blue",lwd=2, lty="dashed")
lines(sort(x.R),predict(forest)[order(x.R)], col="red", lwd=2)


### Now with a deep net ###

### define the network ---------------------------------------------------------

# dimensionality of hidden layer
 d_hidden <- 10

model <- nn_sequential(
  nn_linear(1, d_hidden),
  nn_relu(),
  nn_linear(d_hidden, d_hidden),
  nn_relu(),
  # nn_linear(d_hidden, d_hidden),
  # nn_relu(),
  nn_linear(d_hidden, 1)
)

### network parameters ---------------------------------------------------------

# for adam, need to choose a much higher learning rate in this problem
learning_rate <- 0.001

optimizer <- optim_adam(model$parameters, lr = learning_rate)

### training loop --------------------------------------------------------------

for (t in 1:2000) {
  
  ### -------- Forward pass -------- 
  
  y_pred <- model(x)
  
  ### -------- compute loss -------- 
  loss <- nnf_mse_loss(y_pred, y, reduction = "sum")
  if (t %% 100 == 0)
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

plot(x,y)
lines(sort(x.R),0.5 + sin(10*x.R)[order(x.R)],col="blue",lwd=2, lty="dashed")
lines(sort(as.numeric(x)),y_pred[order(as.numeric(x))],col="red",lwd=2)




