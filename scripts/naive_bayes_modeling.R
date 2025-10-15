library(tidyverse)
library(mice)
library(miceadds)

model_imp <- read_rds(here::here("data/vascular_dementia_indicators.RDS"))
long <- complete(model_imp, "long")
# model_all <- complete(model_imp, "all")

set.seed(101325)
long1 <- long |>
  filter(
    .imp == 1
  ) |>
  select(
    told_high_bp:told_stroke,
    ever_use_coke_heroin_meth:smoke_100cig_life,
    obese
  ) |>
  slice_sample(n = 100)

long1 <-
long1 |>
  mutate(
    dr_told_diabetes_no = case_when(
      dr_told_diabetes == 0 ~ 1,
      TRUE ~ 0
    ),
    dr_told_diabetes_border = case_when(
      dr_told_diabetes == 1 ~ 1,
      TRUE ~ 0
    ),
    dr_told_diabetes_yes = case_when(
      dr_told_diabetes == 2 ~ 1,
      TRUE ~ 0
    ),
    across(
      matches("^dr_told_diabetes"),
      ~as.factor(.x)
    )
  ) |>
  select(
    -dr_told_diabetes
  )

# modeling
library(cmdstanr)
library(posterior)
library(bayesplot)

x_num <- 
long1 |>
  select(
    where(
      is.numeric
    )
  ) |>
  mutate(
    across(
      everything(),
      ~scale(.x)
    )
  )

x_bi <- 
long1 |>
  select(
    where(
      is.factor
    )
  )

stan_list <- list(
  J = nrow(long1), # number of respondents
  C = 2, # number of latent classes (binary: 1 = has VasDem, 0 = doesn't)
  num_feat = ncol(x_num), # number of numeric features
  bi_feat = ncol(x_bi), # number of binary features (includes dummy coded variables)
  x_num = x_num, # data for numeric features/IVs/predictors
  x_bi = x_bi # data for binary (includes dummy coded) features/IVs/predictors
)

nb <- cmdstan_model(here::here("stan/naive_bayes.stan"))

set.seed(101325)
fit <- nb$sample(
  data = stan_list,
  seed = 101325,
  # chains = 4,
  chains = 1,
  # init = 0,
  iter_warmup = 2000,
  iter_sampling = 2000,
  # adapt_delta = .99,
  parallel_chains = parallel::detectCores() - 1
)

fit$output()[[1]]
fit$diagnostic_summary()

summarize_draws(
  fit$draws(),
  default_convergence_measures()
) |> 
  arrange(
    desc(rhat)
  ) |> 
  head()
