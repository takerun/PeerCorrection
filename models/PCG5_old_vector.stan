data{
  int<lower=1> N;
  int<lower=1> uNum;
  int<lower=1> vNum;
  //data=[sender,receiver,grade,corrected]
  int sender[N];
  int receiver[N];
  vector[N] value;
  int corrected[N];
  real hyper[6];
}

parameters{
  vector<lower=0, upper=4>[uNum] ability;
  vector<lower=0.0001>[vNum] reliability;
  vector<lower=-2, upper=2>[vNum] bias;
  real<lower=-2, upper=2> noise;
}

model{
  //prior
  ability ~ normal(hyper[1],1/hyper[2]);
  reliability ~ normal(ability,1/hyper[3]);
  bias ~ normal(0,1/hyper[4]);
  noise ~ normal(0,hyper[5]);
  //posterior
  value ~ normal(ability[receiver]+bias[sender],hyper[6]./reliability[sender]);
  corrected ~ bernoulli(inv_logit(ability[receiver]+bias[sender]+noise));
}
