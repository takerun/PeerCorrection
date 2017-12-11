data{
  int<lower=1> N;
  int<lower=1> uNum;
  int<lower=1> vNum;
  //data=[sender,receiver,corrected]
  int sender[N];
  int receiver[N];
  int corrected[N];
  int senderOrigin[vNum];
  real hyper[4];
}

parameters{
  real ability[uNum];
  real bias[vNum];
  real noise;
}

model{
  //prior
  for(i in 1:uNum){
    ability[i] ~ normal(hyper[1],1/hyper[2]);
  }
  for(i in 1:vNum){
    bias[i] ~ normal(0,1/hyper[3]);
  }
  noise ~ normal(0,hyper[4]);
  //posterior
  for(i in 1:N){
    corrected[i] ~ bernoulli(inv_logit(ability[receiver[i]+1]+bias[sender[i]+1]+noise));
  }
}
