functions {
  real cut_0_1(real value) {
    if (value < 0) {
      return(0);
    }
    if (value > 1) {
      return(1);
    }
    return(value);
  }
}

data{
  int<lower=1> N;
  int<lower=1> uNum;
  int<lower=1> vNum;
  //data=[sender,receiver,corrected]
  int sender[N];
  int receiver[N];
  int corrected[N];
  real hyper[5];
}

parameters{
  real<lower=-2, upper=2> ability[uNum];
  real<lower=-2, upper=2> bias[vNum];
  real<lower=-2, upper=2> noise0;
  real<lower=-1, upper=1> noise1;
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
  noise1 ~ normal(0,hyper[5]);
  //posterior
  for(i in 1:N){
    corrected[i] ~ bernoulli(cut_0_1(inv_logit(ability[receiver[i]+1]+bias[sender[i]+1]+noise0)+noise1));
  }
}
