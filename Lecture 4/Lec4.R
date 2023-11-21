# Introduction to torch based optimization
# Sanjog Misra
# CML, 2022

library(torch)
library(ggplot2)
library(tidyverse)

# Tensors and AD
z <- torch_tensor(.5, requires_grad = TRUE)
y = z^2+z
y$backward()
z$grad


#Optimziation Examples
a <- 1
b <- 5
rosenbrock <- function(x) {
  x1 <- x[1]
  x2 <- x[2]
  (a - x1)^2 + b * (x2 - x1^2)^2
}

df <- expand_grid(x1 = seq(-2, 2, by = 0.01), x2 = seq(-2, 2, by = 0.01)) %>%
  rowwise() %>%
  mutate(x3 = rosenbrock(c(x1, x2))) %>%
  ungroup()

plt = ggplot(data = df,
       aes(x = x1,
           y = x2,
           z = x3)) +
  geom_contour_filled(breaks = as.numeric(torch_logspace(-3, 3, steps = 50)),
                      show.legend = FALSE) +
  theme_minimal() +
  scale_fill_viridis_d(direction = -1) +
  theme(aspect.ratio = 1)

num_iterations <- 1000

# fraction of the gradient to subtract 
lr <- 0.01

# function input (x1,x2)
# this is the tensor w.r.t. which we'll have torch compute the gradient
x_star <- torch_tensor(c(-1, 1), requires_grad = TRUE)
xstor=NULL

for (i in 1:num_iterations) {
  
  if (i %% 100 == 0) cat("Iteration: ", i, "\n")
  
  # call function
  value <- rosenbrock(x_star)
  if (i %% 100 == 0) cat("Value is: ", as.numeric(value), "\n")
  
  # compute gradient of value w.r.t. params
  value$backward()
  if (i %% 100 == 0) cat("Gradient is: ", as.matrix(x_star$grad), "\n\n")
  
  # manual update
  with_no_grad({
    x_star$sub_(lr * x_star$grad)
    x_star$grad$zero_()
  })
  
  xstor = rbind(xstor,c(as.matrix(x_star)))
}


colnames(xstor)=c("x1","x2")
xstor=data.frame(xstor)
xstor$x3 = NA
plt+geom_point(data=data.frame(xstor),aes(x=x1,y=x2),cex=.1,color='red')



# Lets try ADAM
num_iterations <- 1000
x_star <- torch_tensor(c(-1, 1), requires_grad = TRUE)
lr <- .01
optimizer <- optim_adam(x_star, lr)
xstor_adam=NULL
for (i in 1:num_iterations) {
  
  if (i %% 10 == 0) cat("Iteration: ", i, "\n")
  
  optimizer$zero_grad()
  value <- rosenbrock(x_star)
  if (i %% 10 == 0) cat("Value is: ", as.numeric(value), "\n")
  
  value$backward()
  optimizer$step()
  
  if (i %% 10 == 0) cat("Gradient is: ", as.matrix(x_star$grad), "\n\n")
  
  xstor_adam = rbind(xstor_adam,c(as.matrix(x_star)))
  
}

colnames(xstor_adam)=c("x1","x2")
xstor_adam=data.frame(xstor_adam)
xstor_adam$x3 = NA
plt+geom_point(data=data.frame(xstor),aes(x=x1,y=x2),cex=.1,color='red')+geom_point(data=data.frame(xstor_adam),aes(x=x1,y=x2),cex=.1,color='blue')

