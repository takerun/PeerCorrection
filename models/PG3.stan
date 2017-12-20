data{
  int<lower=1> N;
  int<lower=1> uNum;
  int<lower=1> vNum;
  //data=[sender,receiver,grade]
  int sender[N];
  int receiver[N];
  int value[N];
  int senderOrigin[vNum];
  real hyper[5];
}

parameters{
  real<lower=0> ability[uNum];
  real bias[vNum];
}

transformed parameters{
  real<lower=0.00001> reliability[vNum];
  for(i in 1:vNum){
    reliability[i] = hyper[4]*ability[senderOrigin[i]+1]+hyper[3];
  }
}

model{
  //prior
  for(i in 1:uNum){
    ability[i] ~ normal(hyper[1],1/hyper[2]);
  }
  for(i in 1:vNum){
    bias[i] ~ normal(0,1/hyper[5]);
  }
  //posterior
  for(i in 1:N){
    value[i] ~ normal(ability[receiver[i]+1]+bias[sender[i]+1],1/reliability[sender[i]+1]);
  }
}
