## Practical for NHM lab group meeting 19.03.2025

# 1. Load packages & set plot theme ----

# Load packages
pacman::p_load(tidyverse, rstan, brms, bayesplot, tidybayes, shinystan)

# Set plot themes
bayesplot_theme_set(theme_default())
theme_set(bayesplot_theme_get())
color_scheme_set("purple")

# 2. Load & wrangle US states data ----
dat <- cbind(data.frame(state.x77), state.region) %>% 
  rename(region = state.region) %>% 
  rename_with(tolower)

# 3. Run a standard linear model ----
mod <- lm(murder ~ life.exp + illiteracy,  data = dat)

# Visualise how the model fits
par(mfrow = c(1, 2))
plot(dat$murder, predict(mod), xlab = "y", ylab = "Predicted y", main = "lm")

# 4. Run a Bayesian lm using brms::brm() ----
bmod <- brm(formula = murder ~ life.exp + illiteracy,
            data = dat, family = gaussian(),
            chains = 4, cores = 4, iter = 2000)

# Explore structure of model output
print(bmod)  # Check summary of fit, noting Rhat, Bulk_ESS & Tail_ESS values

# Use posterior_predict() to make predictions (replications) using fitted model
yrep <- posterior_predict(bmod)  # analogous to predict()
dim(yrep) # ncol = no. y observations; nrow = no. posterior samples

plot(dat$murder, yrep[sample(1:nrow(yrep), size = 1),], xlab = "y", ylab = "Predicted y", main = "brms (1 sampled posterior value)")

# Plot posteriors & MCMC traces
plot(bmod)  # n=1000 iterations for each MCMC chain (1st 1000 discarded as burn-in)

# Do some posterior predictive checks: do y & yrep look similar?
pp_check(bmod, type = "dens_overlay")  # density overlay
ppc_stat(y = dat$murder, yrep = posterior_predict(bmod), stat = mean) # mean
ppc_stat(y = dat$murder, yrep = posterior_predict(bmod), stat = sd)   # sd
ppc_stat(y = dat$murder, yrep = posterior_predict(bmod), stat = max)  # max
ppc_stat(y = dat$murder, yrep = posterior_predict(bmod), stat = min)  # min - what's wrong here?

# Reformulate model to truncate predicted murder rate at 0
bmod_trunc <- brm(formula = murder | trunc(lb = 0) ~ life.exp + illiteracy,
                  data = dat, family = gaussian(),
                  chains = 4, cores = 4, iter = 2000)

# Check minimum predicted murder rate is now consistent with observations
ppc_stat(y = dat$murder, yrep = posterior_predict(bmod_trunc), stat = min)

# But are we capturing all the important variation? Posterior predictive checks by region
p1 <- bayesplot_grid(  
  ppc_stat_grouped(y = dat$murder, yrep = posterior_predict(bmod_trunc), stat = mean, group = dat$region),
  ppc_stat_grouped(y = dat$murder, yrep = posterior_predict(bmod_trunc), stat = sd, group = dat$region),
  ppc_stat_grouped(y = dat$murder, yrep = posterior_predict(bmod_trunc), stat = min, group = dat$region),
  ppc_stat_grouped(y = dat$murder, yrep = posterior_predict(bmod_trunc), stat = max, group = dat$region)
)
# p1

# Summarise mean murder rate by region
dat %>% group_by(region) %>% reframe(mean_murder_rate = mean(murder))

# Reformulate model to include region
bmod_trun_reg <- brm(formula = murder | trunc(lb = 0) ~ life.exp + illiteracy + region,
                     data = dat, family = gaussian(),
                     chains = 4, cores = 4, iter = 2000)

# Check visually: is model fit improved?
color_scheme_set("blue")  # distinguish new model with diff colour
p2 <- bayesplot_grid(  # PPCs for regional model
  ppc_stat_grouped(y = dat$murder, yrep = posterior_predict(bmod_trun_reg), stat = mean, group = dat$region),
  ppc_stat_grouped(y = dat$murder, yrep = posterior_predict(bmod_trun_reg), stat = sd, group = dat$region),
  ppc_stat_grouped(y = dat$murder, yrep = posterior_predict(bmod_trun_reg), stat = min, group = dat$region),
  ppc_stat_grouped(y = dat$murder, yrep = posterior_predict(bmod_trun_reg), stat = max, group = dat$region)
)
# p2
color_scheme_set("purple")  # reset colour scheme

patchwork::wrap_plots(p1, p2, nrow = 2)  # visually compare the 2 models

# Compare models more formally using approximate leave one out cross validation
loo1 <- loo(bmod_trunc, cores = 4)    # model with 0-truncated response
loo2 <- loo(bmod_trun_reg, cores = 4) # 0-truncated response & region

loo_compare(loo1, loo2)  # elpd_diff smaller = better fit for regional model

# 5. Summarise favoured model ----

# Make some plots with built-in functions
mcmc_trace(x = bmod_trun_reg)  # MCMC traces
mcmc_pairs(x = bmod_trun_reg)  # Parameter correlations (NB can be long to run)

# Inspect the fit interactively using shinystan::launch_shinystan()
my_sso <- launch_shinystan(bmod_trun_reg)

# Extract posterior using tidy_bayes::tidy_draws()
tidy_post <- bmod_trun_reg %>% tidy_draws() 

# Plot posterior estimates of each parameter
tidy_post %>% 
  select(draw = .draw, contains(colnames(dat))) %>% # get parameters
  pivot_longer(!draw, names_to = "pars") %>% 
  group_by(pars) %>% 
  reframe(  # Calculate medians & 95% credible intervals
    pars,
    med = median(value),
    lower = quantile(value, 0.025),
    upper = quantile(value, 0.975)
  ) %>% 
  ggplot(aes(x = pars)) + 
  geom_point(aes(y = med)) + 
  geom_errorbar(aes(ymin = lower, ymax = upper), width = .2) + 
  geom_hline(aes(yintercept = 0), linetype = 2, col = "grey") +
  labs(x = "Parameter", y = "Posterior estimate")
  
# Plot observations against predictions for some chosen predictor variable
ppc_intervals_grouped(x = dat$life.exp, y = dat$murder, yrep = posterior_predict(bmod_trun_reg), group = dat$region)

# 6. Use your own custom Stan model ----

# Compile custom Stan model
stan_mod <- stan_model(file = "murder_mod.stan")

# Make list of data for model
stan_dat <- list(
  N = nrow(dat), 
  y = dat$murder,
  P = 3,
  X = dat %>% select(life.exp, illiteracy, region)
)

# Fit Stan model (NB not zero-truncated response)
stan_fit <- sampling(object = stan_mod, data = stan_dat, iter = 2000, 
                     chains = 4, cores = 4, seed = 007, verbose = TRUE)

# Summarise fit
stan_fit  # Ensure R_hat = 1 & n_eff is high (~ >1000)

# Extract predicted response
yrep <- extract(stan_fit, "y_rep")$y_rep

# Do posterior predictive checks
bayesplot_grid(
  ppc_stat_grouped(y = dat$murder, yrep = yrep, stat = mean, group = dat$region),
  ppc_stat_grouped(y = dat$murder, yrep = yrep, stat = sd, group = dat$region),
  ppc_stat_grouped(y = dat$murder, yrep = yrep, stat = min, group = dat$region),
  ppc_stat_grouped(y = dat$murder, yrep = yrep, stat = max, group = dat$region)
)

ppc_intervals_grouped(x = dat$life.exp, y = dat$murder, yrep = yrep, group = dat$region)

# Inspect the fit interactively using shinystan::launch_shinystan()
my_sso <- launch_shinystan(stan_fit)
