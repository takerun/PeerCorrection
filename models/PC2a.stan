data{
  int<lower=1> N;
  int<lower=1> uNum;
  int<lower=1> vNum;
  //data=[sender,receiver,diff]
  int sender[N];
  int receiver[N];
  int diff[N];
  int senderOrigin[vNum];
  real hyper[5];
}

parameters{
  real<lower=0> ability[uNum];
  real bias[vNum];
  real noise0;
}

model{
  //prior
  for(i in 1:uNum){
    ability[i] ~ normal(hyper[1],1/hyper[2]);
  }
  for(i in 1:vNum){
    bias[i] ~ normal(0,1/hyper[3]);
  }
  noise0 ~ normal(0,hyper[4]);
  //posterior
  for(i in 1:N){
    diff[i] ~ poisson(1/(ability[receiver[i]+1]+bias[sender[i]+1]))+noise0;
  }
}
