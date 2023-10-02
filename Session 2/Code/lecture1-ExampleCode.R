# Causal ML
# Max Farrell & Sanjog Misra
# Fall 2023
# Session 1

# Simple Examples
N=1000
set.seed(1212)
err = rnorm(N)
x = rnorm(N)
prb =  1/(1+exp(-x))
prb=.5
trt = rbinom(N,1,prb)
y = 10-trt+2*x+.25*err
summary(lm(y~trt))
summary(lm(y~trt+x))

# difference in Means
tau = mean(y[trt==1])-mean(y[trt==0])
tt = t.test(y[trt==1],y[trt==0])
tt$conf.int

# Regression
r1 = lm(y~trt)
confint(r1)
r2 = lm(y~trt+x)
confint(r2)

# Simple (no X) Parametric Model
N=10000
set.seed(1234)
trt = rbinom(N,1,0.5)
e0 = rnorm(N)
e1 = rnorm(N)
a0 = a1 = -1
sig0 = 1
sig1 = 2
tau = 2
y0=exp(a0+sig0*e0)
y1=exp(a1+tau*trt+sig1*e1)
y = trt*y1+(1-trt)*y0
lny = round(log(y),4)
tru.ate = exp(a1+tau+.5*sig1^2)-exp(a0+.5*sig0^2)

# Negative log likelihood
nloglik = function(theta=theta0){
  xb0 = theta[1]
  xb1 = theta[2]
  -sum(
    (1-trt)*dnorm(lny-xb0,0,theta[3],log=TRUE)+
      (trt)*dnorm(lny-xb1,0,theta[4],log=TRUE))
}

theta0 = c(0,0,1,1)
o1 = suppressWarnings(optim(par = theta0,fn = nloglik,method = "BFGS",hessian = TRUE))
theta = o1$par

# Treatment effect
mu0 = exp(theta[1]+.5*theta[3]^2)
mu1 = exp(theta[2]+.5*theta[4]^2)
tau = mu1-mu0;tau


# Delta Method
# derivatives
g0 = c(mu0,0,mu0*theta[3],0)
g1 = c(0,mu1,0,mu1*theta[4])
g = g1-g0
g=matrix(g,4,1)
vc = (solve(o1$hessian))
# SE
se.tau = sqrt(t(g)%*%vc%*%g)
se.tau

# Difference in Means
mean(y1)/mean(trt)-mean(y0)/(mean(1-trt))

# Regression
summary(lm(y~trt))

# Check
library(msm)
deltamethod(~(exp(x2+.5*x4^2)-exp(x1+.5*x3^2)),theta,vc)

library(stats4)
m1 = mle(nloglik,start = theta)
coef(summary(m1))



