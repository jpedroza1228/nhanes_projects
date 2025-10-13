library(tidyverse)
library(mice)
library(miceadds)

# this dataset needs to be cleaned
# drop values with 7 (e.g., 77)
# make values with 9 (e.g., 99) missing for imputation
# reverse binary from 1 = Yes, 2 = No to 1 = Yes, 0 = No
# final indicators may want to be categorical for ease of use in naive bayes stan model
# imputation model should have ~30 imputation with 10-15 iterations per
# latino <- read_csv("latino_vascular_dementia_indicators.csv")

latino <- read_csv("https://raw.githubusercontent.com/jpedroza1228/nhanes_projects/refs/heads/main/latino_vascular_dementia_indicators.csv")

nrow(latino)

model_df <- latino |> 
  filter(
    cerad_complete_status == 1 &
    animal_fluency_sample_test == 1 &
    digit_symbol_sample_test == 1
   ) |>
  select(
    c(
      sex,
      age,
      # race_ethnic,
      # birth_country,
      citizen,
      # length_us,
      ed,
      # marital,
      # annual_house_income,
      fam_income_pov_ratio,
      # total_num_house,
      # total_num_fam,
      told_high_bp, # high blood pressure risk factor
      told_high_bp_2plus, # high blood pressure
      dr_told_high_chol, # high cholesterol
      cerad_score_trial1_recall, # cognitive risk factor
      cerad_score_trial2_recall, # cognitive
      cerad_score_trial3_recall, # cognitive
      cerad_score_delay_recall, # cognitive 
      cerad_intrusion_wordcount_trial1, # cognitive
      cerad_intrusion_wordcount_trial2, # cognitive
      cerad_intrusion_wordcount_trial3, # cognitive
      cerad_intrusion_wordcount_recall, # cognitive
      animal_fluency_score, # cognitive
      digit_symbol_score, # cognitive
      diff_think_remember, # cognitive
      dr_told_diabetes, # diabetes risk factor
      told_prediabetes, # diabetes
      told_risk_diabetes, # diabetes
      feel_risk_diabetes, # diabetes
      told_heart_fail, # heart risk factor
      told_heart_disease, # heart
      told_angina, # heart
      told_heart_attack, # heart
      told_stroke, # stroke risk factor
      dr_told_overweight, # obesity risk factor
      dr_told_lose_wt, # obesity 
      ever_use_coke_heroin_meth, # drug use risk factor
      ever_45_drink_everyday, #heavy drinking risk factor
      num_ready_eat_30day, # unhealthy eating
      num_frozen_meal_30day, # unhealthy eating
      dr_told_exercise, # pa risk factor
      walk_bike, # pa
      vig_rec_act, # pa
      mod_rec_act, # pa
      dr_told_sleep_trouble, # sleep risk factor
      dr_told_sleep_disorder, # sleep
      smoke_100cig_life, # cig smoking risk factor
      bmi
    )
   )


model_df <- model_df |> 
  filter(
    citizen != 7 &
    ed != 7
  ) |>
  mutate(
    across(
      c(
        told_high_bp,
        told_high_bp_2plus,
        dr_told_high_chol,
        told_prediabetes,
        told_heart_fail,
        told_heart_disease,
        told_angina,
        told_heart_attack,
        told_stroke,
        dr_told_overweight,
        ever_use_coke_heroin_meth,
        ever_45_drink_everyday,
        walk_bike,
        vig_rec_act,
        mod_rec_act,
        told_risk_diabetes,
        feel_risk_diabetes,
        dr_told_sleep_trouble,
        dr_told_sleep_disorder,
        smoke_100cig_life,
        diff_think_remember
      ),
      ~case_when(
        .x == 1 ~ 1,
        .x == 2 ~ 0,
        TRUE ~ NA_integer_
      )
    ),
    across(
      c(
        dr_told_diabetes
      ),
      ~case_when(
        .x == 1 ~ 2,
        .x == 2 ~ 0,
        .x == 3 ~ 1,
        TRUE ~ NA_integer_
      )
    ),
    female = case_when(
      sex == 1 ~ 0,
      sex == 2 ~ 1
    ),
    citizen = case_when(
      citizen == 1 ~ 1,
      citizen == 2 ~ 0,
      TRUE ~ NA_integer_
    ),
     ed = case_when(
      ed == 9 ~ NA_integer_,
      is.null(ed) ~ NA_integer_,
      TRUE ~ ed
    ),
    obese = case_when(
      bmi < 18.5 ~ 0,
      bmi >= 18.5 & bmi < 25 ~ 0,
      bmi >= 25 & bmi < 30 ~ 0,
      bmi >= 30 ~ 1,
      is.null(bmi) ~ NA_integer_
    )
  )

model_df <- model_df |> 
  select(
    -c(
      bmi,
      sex,
      dr_told_lose_wt,
      dr_told_exercise
    )
  )

model_df <- 
model_df |> 
  mutate(
    across(
      c(
        citizen,
        ed,
        told_high_bp,
        told_high_bp_2plus,
        dr_told_high_chol,
        dr_told_diabetes,
        told_prediabetes,
        told_risk_diabetes,
        feel_risk_diabetes,
        told_heart_fail,
        told_heart_disease,
        told_angina,
        told_heart_attack,
        told_stroke,
        dr_told_overweight,
        ever_use_coke_heroin_meth,
        ever_45_drink_everyday,
        walk_bike,
        vig_rec_act,
        mod_rec_act,
        dr_told_sleep_trouble,
        dr_told_sleep_disorder,
        smoke_100cig_life,
        female,
        obese,
        diff_think_remember
      ),
      ~as.factor(.x)
    )
  )

glimpse(model_df)


pred_matrix <- make.predictorMatrix(data = model_df)
imp_method <- make.method(data = model_df)

num_columns <- c("num_frozen_meal_30day", "num_ready_eat_30day",
"digit_symbol_score", "animal_fluency_score", "cerad_intrusion_wordcount_recall",
"cerad_intrusion_wordcount_trial3", "cerad_intrusion_wordcount_trial2",
"cerad_intrusion_wordcount_trial1", "cerad_score_delay_recall",
"cerad_score_trial3_recall", "cerad_score_trial2_recall",
"cerad_score_trial1_recall", "fam_income_pov_ratio", "age")
bi_columns <- c("obese", "female", "smoke_100cig_life",
"dr_told_sleep_disorder", "dr_told_sleep_trouble",
"mod_rec_act", "vig_rec_act", "walk_bike",
"ever_45_drink_everyday", "ever_use_coke_heroin_meth",
"dr_told_overweight", "told_stroke", "told_heart_attack",
"told_angina", "told_heart_disease", "told_heart_fail",
"feel_risk_diabetes", "told_risk_diabetes", "told_prediabetes",
"diff_think_remember", "dr_told_high_chol", "told_high_bp_2plus",
"told_high_bp", "citizen")
cat_columns <- c("dr_told_diabetes", "ed")

imp_method[num_columns] <- "pmm"
imp_method[bi_columns] <- "logreg.boot"
imp_method[cat_columns] <- "polyreg"

# pred_matrix[, "seqn"] <- 0
# pred_matrix[, "birth_country"] <- 0
pred_matrix[c("told_prediabetes",
              "feel_risk_diabetes",
              "told_risk_diabetes"), "dr_told_diabetes"] <- 0
pred_matrix["told_high_bp_2plus", "told_high_bp"] <- 0

prac <- 
  mice(
  model_df,
  maxit = 1,
  m = 1,
  method = imp_method,
  predictorMatrix = pred_matrix
)

prac$loggedEvents

rm(prac)
gc()

inspectdf::inspect_na(model_df)

set.seed(101125)
model_imp <- 
  mice(
    model_df,
    m = 50,
    maxit = 20,
    method = imp_method,
    predictorMatrix = pred_matrix
  )

model_imp$loggedEvents

# saveRDS(model_imp, here::here("projects/nhanes/imputation_model_more_var.rds"))