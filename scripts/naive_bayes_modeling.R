library(tidyverse)
library(mice)
library(miceadds)
library(tidyLPA)
library(poLCA)

select <- dplyr::select

model_imp <- read_rds(here::here("data/vascular_dementia_indicators.RDS"))
long <- complete(model_imp, "long")
# model_all <- complete(model_imp, "all")

# write.csv(long, "vascular_dementia_indicators_long.csv")

purrr::map2(
  long |>
  filter(
    .imp == 1
  ) |>
  select(
    cerad_score_trial1_recall:digit_symbol_score,
    num_ready_eat_30day:num_frozen_meal_30day
  ),
  long |>
  filter(
    .imp == 1
  ) |>
  select(
    cerad_score_trial1_recall:digit_symbol_score,
    num_ready_eat_30day:num_frozen_meal_30day
  ) |>
  colnames(),
  ~ggplot(
    data = long |>
    filter(.imp == 1),
    aes(
      .x
    )
  ) +
  geom_histogram(
    color = "black",
    fill = "seagreen",
    bins = 15
  ) +
  labs(title = glue::glue("{.y}")) +
  theme_light()
  )



# 1 imputation
long1 <-
long |>
filter(
    .imp == 1
  ) |>
  select(
    told_high_bp:told_stroke,
    ever_use_coke_heroin_meth:smoke_100cig_life,
    obese
  )

# only 50 participants for MCMC
set.seed(101925)
long1_sub <-
long1 |>
  slice_sample(n = 50)

long_num <-
# long1 |>
long1_sub |>
  select(
    cerad_score_trial1_recall:digit_symbol_score
    # num_ready_eat_30day:num_frozen_meal_30day
  )

long_bi <- 
# long1 |>
long1_sub |>
mutate(
    dr_told_diabetes_yes = case_when(
      dr_told_diabetes %in% c(1, 2) ~ 1,
      TRUE ~ 0
    )
  ) |>
  select(
    -dr_told_diabetes
  ) |>
  select(
    told_high_bp:dr_told_high_chol,
    diff_think_remember:ever_use_coke_heroin_meth,
    walk_bike:dr_told_diabetes_yes
  ) |>
  mutate(
    across(
      where(is.factor),
      ~as.numeric(.x)
    ),
    across(
      -matches("dr_told_diabetes"),
      ~case_when(
        .x == 2 ~ 1,
        .x == 1 ~ 0
      )
    )
  )


# factor analysis
# set.seed(101925)
# fa_model <- 
# psych::fa(
#   long_num,
#   nfactors = 3,
#   fm = "ml",
#   rotate = "oblimin",
#   scores = "regression"
#   )
# fa_model
# fa_model$scores

set.seed(102125)
pca <- map(
  2:6,
  ~psych::principal(
    long_num,
    nfactors = .x,
    method = "regression",
    rotate = "varimax"
  )
)

tibble(
  model = c(2, 3, 4, 5, 6),
  fit = map_dbl(pca, ~.x$fit),
  rms = map_dbl(pca, ~.x$rms)
)

pca[[1]]$loadings
pca[[1]]$scores


# purrr::map(
#   long_bi |>
#   select(
#     told_high_bp,
#     told_high_bp_2plus,
#     dr_told_high_chol,
#     diff_think_remember,
#     told_prediabetes,
#     told_risk_diabetes,
#     feel_risk_diabetes,
#     told_heart_fail,
#     told_heart_disease,
#     told_angina,
#     told_heart_attack,
#     told_stroke,
#     ever_use_coke_heroin_meth,
#     walk_bike,
#     vig_rec_act,
#     mod_rec_act,
#     dr_told_sleep_trouble,
#     dr_told_sleep_disorder,
#     smoke_100cig_life,
#     dr_told_diabetes_yes
#   ),
#   ~count(data.frame(x = .x), x)
# )

# long_bi |>
#   select(
#     told_high_bp,
#     told_high_bp_2plus,
#     dr_told_high_chol,
#     diff_think_remember,
#     told_prediabetes,
#     told_risk_diabetes,
#     feel_risk_diabetes,
#     told_heart_fail,
#     told_heart_disease,
#     told_angina,
#     told_heart_attack,
#     told_stroke,
#     ever_use_coke_heroin_meth,
#     walk_bike,
#     vig_rec_act,
#     mod_rec_act,
#     dr_told_sleep_trouble,
#     dr_told_sleep_disorder,
#     smoke_100cig_life,
#     dr_told_diabetes_yes
#   ) |>
#   cor() |>
#   round(2) |>
#   as_tibble() |>
#   mutate(
#     row_names = long_bi |>
#     select(
#       told_high_bp,
#     told_high_bp_2plus,
#     dr_told_high_chol,
#     diff_think_remember,
#     told_prediabetes,
#     told_risk_diabetes,
#     feel_risk_diabetes,
#     told_heart_fail,
#     told_heart_disease,
#     told_angina,
#     told_heart_attack,
#     told_stroke,
#     ever_use_coke_heroin_meth,
#     walk_bike,
#     vig_rec_act,
#     mod_rec_act,
#     dr_told_sleep_trouble,
#     dr_told_sleep_disorder,
#     smoke_100cig_life,
#     dr_told_diabetes_yes
#     ) |>
#     colnames()
#   ) |>
#   relocate(
#     row_names,
#     .before = 1
#   ) |>
#   gt::gt()

# f <- with(
#   long_bi,
#   cbind(
#     told_high_bp,
#     told_high_bp_2plus,
#     dr_told_high_chol,
#     diff_think_remember,
#     # told_prediabetes,
#     # told_risk_diabetes,
#     # feel_risk_diabetes,
#     told_heart_fail,
#     told_heart_disease,
#     # told_angina,
#     told_heart_attack,
#     told_stroke,
#     ever_use_coke_heroin_meth,
#     # walk_bike,
#     vig_rec_act,
#     mod_rec_act,
#     dr_told_sleep_trouble,
#     dr_told_sleep_disorder,
#     smoke_100cig_life,
#     dr_told_diabetes_yes
# ) ~ 1
# )

# set.seed(101925)
# lca_mod <- poLCA(
#   f,
#   long_bi,
#   nclass = 2,
#   nrep = 10
#   )

# lca_mod



# MCMC Sampling
library(cmdstanr)
library(posterior)
library(bayesplot)

long_num_sub <- 
tibble(
  factor1 = pca[[1]]$scores[,1],
  factor2 = pca[[2]]$scores[,2]
)

long_bi <- long_bi |>
  select(
    told_high_bp,
    told_high_bp_2plus,
    dr_told_high_chol,
    diff_think_remember,
    told_heart_fail,
    told_heart_disease,
    told_heart_attack,
    told_stroke,
    ever_use_coke_heroin_meth,
    vig_rec_act,
    mod_rec_act,
    dr_told_sleep_trouble,
    dr_told_sleep_disorder,
    smoke_100cig_life,
    dr_told_diabetes_yes
  )

stan_list <- list(
  J = nrow(long_bi), # number of respondents
  C = 2, # number of latent classes (binary: 1 = has VasDem, 0 = doesn't)
  num_feat = ncol(long_num_sub), # number of numeric features
  bi_feat = ncol(long_bi), # number of binary features (includes dummy coded variables)
  order_feat = 1,
  x_num = long_num_sub, # data for numeric features/IVs/predictors
  x_bi = long_bi # data for binary (includes dummy coded) features/IVs/predictors
)

rm(nb, fit)
gc()

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

# fit$output()[[1]]
fit$diagnostic_summary()

summarize_draws(
  fit$draws(),
  default_convergence_measures()
) |> 
  arrange(
    desc(rhat)
  )
  # filter(rhat > 1.1) |>
  # gt::gt()

nb_measure <- summarize_draws(fit$draws(), default_summary_measures())

nb_measure |>
  filter(
    str_detect(
      variable,
      "nu"
    )
  )

nb_measure |>
  filter(
    str_detect(
      variable,
      "^mu\\[1,"
    )
  )

nb_measure |>
  filter(
    str_detect(
      variable,
      "^sigma\\[1,"
    )
  )

log_nu <- fit$draws("log_nu") |> as_draws_matrix() |> janitor::clean_names()

mcmc_trace(log_nu)
mcmc_dens(exp(log_nu))
mcmc_intervals(exp(log_nu))


log_nu |>
  as_tibble() |>
  mutate(
    across(
      everything(),
      ~as.numeric(.x)
    )
  ) |>
  pivot_longer(
    everything()
  ) |>
  ggplot(
    aes(
      name,
      exp(value)
    )
  ) +
  geom_boxplot()
