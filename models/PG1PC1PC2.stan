data{
  int<lower=1> N;
  int<lower=1> uNum;
  int<lower=1> vNum;
  //data=[sender,receiver,grade,corrected,diff]
  int sender[N];
  int receiver[N];
  int value[N];
  int corrected[N];
  int diff[N];
  int senderOrigin[vNum];
  real hyper[7];
}

parameters{
  real ability[uNum];
  real<lower=0.00001> reliability[vNum];
  real bias[vNum];
  real noise0;
  real noise1;
}

model{
  //prior
  for(i in 1:uNum){
    ability[i] ~ normal(hyper[1],1/hyper[2]);
  }
  for(i in 1:vNum){
    reliability[i] ~ gamma(hyper[3],hyper[4]);
    bias[i] ~ normal(0,1/hyper[5]);
  }
  noise0 ~ normal(0,hyper[6]);
  noise1 ~ normal(0,hyper[7]);
  //posterior
  for(i in 1:N){
    value[i] ~ normal(ability[receiver[i]+1]+bias[sender[i]+1],1/reliability[sender[i]+1]);
    corrected[i] ~ bernoulli(inv_logit(ability[receiver[i]+1]+bias[sender[i]+1]+noise0));
    diff[i] ~ poisson(exp(-(ability[receiver[i]+1]+bias[sender[i]+1]+noise1)));
  }
}
