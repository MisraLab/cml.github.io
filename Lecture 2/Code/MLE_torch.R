# Code to Understand basic torch implementations of MLE
# Max Farrell & Sanjog Misra
# CML, 2023

library(torch)
set.seed(1234)
n_obs <- 1000
x <- runif(n_obs * 2) %>% matrix(n_obs, 2)
beta_true <- matrix(c(1, 1.8), nrow=2, ncol=1)
lambda_true <- exp(x %*% beta_true)
summary(lambda_true)
y <- rpois(n_obs, lambda_true)
summary(y)

hist(y)

tvars <- list()
tvars$x <- torch_tensor(x)
tvars$y <- torch_tensor(y)

LogLik <- function(beta, tvars) {
  # if (!is(beta, "torch_tensor")) {
  #   stop("beta must be a torch tensor")
  # }
  log_lambda <- torch_matmul(tvars$x, beta)
  lp <- torch_sum(tvars$y * log_lambda - torch_exp(log_lambda))
  lp
}

# Sanity check that it works
LogLik(torch_tensor(beta_true), tvars)


LogLikGrad <- function(beta) {
  beta_ad <- torch_tensor(beta, requires_grad=TRUE)
  loss <- LogLik(beta_ad, tvars)
  grad <- autograd_grad(loss, beta_ad)[[1]]
  (as.numeric(grad))
}

# Just check that this runs and has the correct dimensions
LogLikGrad(beta_true)

# From now on we'll take `tvars` to be a global variable to save
# writing everything as lambda functions.

EvalLogLik <- function(beta) {
  log_lik <- LogLik(torch_tensor(beta), tvars) %>% as.numeric()
  log_lik
}

# Just check that this runs
EvalLogLik(beta_true)
optim_time <- Sys.time()
opt_result <- optim(
  fn=EvalLogLik,
  gr=LogLikGrad,
  method="BFGS",
  par=c(0, 0),
  control=list(fnscale=-1 / n_obs))
optim_time <- Sys.time() - optim_time

data.frame("Estimate"=opt_result$par, "Truth"=beta_true) %>% print()

# Gradient
LogLikGrad(opt_result$par)
beta_hat <- opt_result$par


glm_time <- Sys.time()
glm_fit <- glm(
  y ~ x1 + x2 - 1,
  data=data.frame(y=y, x1=x[, 1], x2=x[, 2]),
  start=c(0, 0),
  family="poisson")
glm_time <- Sys.time() - glm_time

max(abs(coefficients(glm_fit) - beta_hat))

optim_time
glm_time

LogLikGrad(coefficients(glm_fit))


LogLikHessian <- function(beta) {
  beta_ad <- torch_tensor(beta, requires_grad=TRUE)
  log_lik <- LogLik(beta_ad, tvars)
  # The argument `create_graph` allows `grad` to be itself differentiated, and
  # the argument `retain_graph` saves gradient computations to make repeated differentiation
  # of the same quantity more efficient.
  grad <- autograd_grad(log_lik, beta_ad, retain_graph=TRUE, create_graph=TRUE)[[1]]
  
  # Now we compute the gradient of each element of the gradient, each of which is
  # one row of the Hessian matrix.
  hess <- matrix(NA, length(beta), length(beta))
  for (d in 1:length(grad)) {
    hess[d, ] <- autograd_grad(grad[d], beta_ad, retain_graph=TRUE)[[1]] %>% as.numeric()
  }
  return(hess)
}

# Just check that this runs and has the correct dimensions
LogLikHessian(beta_true)

fisher_info <- -1 * LogLikHessian(opt_result$par)

torch_se <- sqrt(diag(solve(fisher_info)))
glm_se <- summary(glm_fit)$coefficients[, "Std. Error"]

rbind(torch_se,glm_se)




