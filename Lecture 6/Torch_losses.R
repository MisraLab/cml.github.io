
# Various loss functions

# Our own Cross-entropy (binary -logLike)
binary_nll = function(util,y){
    probs = torch_sigmoid(util)
    ll = y*torch_log(probs)+(1-y)*torch_log(1-probs)    
    nll = -torch_sum(ll)
    nll
}

# Allow for probit
binary_nll2 = function(util,y,model='logit'){
  if(model=='logit'){ probs = torch_sigmoid(util) }
  if(model=='probit'){ 
    z = distr_normal(0,1)
    probs = z$cdf(util)
    probs = torch_clip(probs,0.00001,1-0.00001) # clip extremes
    }
  ll = y*torch_log(probs)+(1-y)*torch_log(1-probs)    
  nll = -torch_sum(ll)
  nll
}

# Let's code up the Tobit
tobit_nll = function(yhat,y,sig){
  z = distr_normal(0,sig)
  prob0 = z$cdf(-yhat)
  prob0 = torch_clip(probs,0.00001,1-0.00001)
  ind = 0+1*(y==0)
  err = y-yhat
  ll = ind*torch_log(prob0)+(1-ind)*z$log_prob(err)
  nll = -torch_mean(ll)
  nll
}




