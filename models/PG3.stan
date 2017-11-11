data{
  int<lower=1> N;
  int<lower=1> uNum;
  int<lower=1> vNum;
  //data=[sender,receiver,grade]
  int sender[N];
  int receiver[N];
  int value[N];
  real hyper[5];
}

parameters{
  real<lower=0, upper=4> ability[uNum];
  real<lower=-2, upper=2> bias[vNum];
}

transformed parameters{
  real<lower=0.0001> reliability[vNum];
  for(i in 1:uNum){
    reliability[i] = hyper[4]*ability[i]+hyper[3];
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
