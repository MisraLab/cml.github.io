# A simple "application"
# CML 2023
# Max Farrell & Sanjog Misra

# Simulate Data
set.seed(1234)
n <- 1000
# Covariates
k <- 10
xtru <- matrix(runif(n*k),n,k)
# Parameter functions
btru = 1+xtru[,1:2]%*%c(-1,2)
atru = 3+xtru[,3:4]%*%c(1,-.5)
# Propensity 
prb = 1/(1+exp(-1-2*xtru[,1]+(xtru[,2])))
summary(prb)

# Generate Treatment
trt = rbinom(n,1,prb)
# trt = rbinom(n,1,.5)
# Outcomes
ytru = atru+ btru*trt + rnorm(n)

# Plots
# mu0 = E[Y|trt=0]
plot(density(atru))
# True CATEs
plot(density(btru),main="",xlab="CATE")
# True ATE
mean(btru)

# Regression
dat = data.frame(xtru,trt,ytru)
stor = NULL
for(b in 1:100){
smp = sample(1:n,n,replace = TRUE)  
res = lm(ytru~trt,data=dat[smp,])
stor = rbind(stor,coef(res))
}

# Compare to regression
summary(lm(ytru~trt))
sqrt(var(stor[,2]))

# Adding covariates
# What does this do?
summary(lm(ytru~trt+xtru))

# Regression with interactions 
res = lm(ytru~trt*xtru,data=dat)

# propensity Score
gres = glm(trt~xtru,data=dat,family='binomial')
phat = predict(gres,type='response')
plot(phat~prb)
abline(0,1)

# Predict Y1
df1 = dat
df1$trt=1
yhat1 = predict(res,newdata=df1)

# Predict Y0
df0 = dat
df0$trt=0
yhat0 = predict(res,newdata=df0)


# CATEs
tau = (yhat1-yhat0)
# True CATEs
plot(density(btru),main="",xlab="CATE",ylim=c(0,1))
lines(density(tau),lty=2)

# Transformed Outcomes
ty = ((trt - phat)/(phat*(1-phat)))*ytru
mean(ty)
sqrt(var(ty))

# DR Estimator
dr.cate = (yhat1+trt*(ytru-yhat1)/phat)-(yhat0+(1-trt)*(ytru-yhat0)/(1-phat))
dr.tau = mean(yhat1+trt*(ytru-yhat1)/phat)-mean(yhat0+(1-trt)*(ytru-yhat0)/(1-phat))
dr.tau
sqrt(var(dr.cate)/n)


# (Counterfactual) Policy Evaluation
cost = 1
# Policy
trt_star = 0+1*(tau>cost)
trt_tru =  0+1*(btru>cost)
table(trt_star,trt_tru)

# Profit
df1 = dat
df1$trt=trt_star
prft_star = sum(predict(res,newdata=df1) - cost*trt_star) 
prft_star

df1 = dat
df1$trt=as.numeric(trt_tru)
prft_tru = sum(predict(res,newdata=df1) - cost*trt_tru) 
prft_tru

# Counterfactual Profits
sum((trt_star*trt/phat)*(ytru - cost) + ((1-trt_star)*(1-trt)/(1-phat))*(ytru))
sum((trt_tru*trt/phat)*(ytru - cost) + ((1-trt_tru)*(1-trt)/(1-phat))*(ytru))

