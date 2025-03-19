// ** Example Stan model to predict US murder rates **

// Define the input data
data {
  int<lower=0> N;  // No. observations
  vector[N] y;     // Response variable (y)
  int<lower=0> P;  // No. predictors (x)
  matrix[N, P] X;  // Covariate matrix
}

// Define parameters accepted by the model
parameters {
  // Intercept
  real alpha;
  // Vector of coefficients for covariates
  vector[P] beta;
  // Variance
  real<lower=0> sigma;
}

// Define priors & model
model {
  // Define priors
  alpha ~ normal(100, 25);
  beta  ~ normal(0, 2);
  sigma  ~ normal(0, 2);
  
  // Define model to be estimated (NB Stan vectorises the `X * beta`)
  y ~ normal(alpha + X * beta, sigma);
}

// Define other things you want to be returned
generated quantities {
  // Posterior predictive distribution (for posterior predictive checks)
  array[N] real y_rep = normal_rng(alpha + X * beta, sigma);
}
