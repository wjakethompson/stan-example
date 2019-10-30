data {
  int<lower=1> I;                            // number of items
  int<lower=1> J;                            // number of respondents
  int<lower=1> N;                            // number of observations
  int<lower=1, upper=I> ii[N];               // item of observation n
  int<lower=1, upper=J> jj[N];               // respondent of  observation n
  int<lower=0, upper=1> y[N];                // score of observation n
}
parameters {
  vector[I] b;
  vector<lower=0>[I] a;
  vector<lower=0, upper=1>[I] c;
  
  vector[J] theta;
}
model {
  vector[N] eta;
  
  // priors
  b ~ normal(0, 10);
  a ~ lognormal(0, 1);
  c ~ beta(5, 17);
  theta ~ normal(0, 1);
  
  for (n in 1:N) {
    eta[n] = c[ii[n]] + (1 - c[ii[n]]) * inv_logit(a[ii[n]] * (theta[jj[n]] - b[ii[n]]));
  }
  
  y ~ bernoulli(eta);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_lpmf(y[n] | c[ii[n]] + (1 - c[ii[n]]) * inv_logit(a[ii[n]] * (theta[jj[n]] - b[ii[n]])));
  }
}
