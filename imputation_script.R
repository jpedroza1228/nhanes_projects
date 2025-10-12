library(tidyverse)
library(mice)
library(miceadds)

# this dataset needs to be cleaned
# drop values with 7 (e.g., 77)
# make values with 9 (e.g., 99) missing for imputation
# reverse binary from 1 = Yes, 2 = No to 1 = Yes, 0 = No
# final indicators may want to be categorical for ease of use in naive bayes stan model
# imputation model should have ~30 imputation with 10-15 iterations per
latino <- read_csv("latino_vascular_dementia_indicators.csv")

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
    num_ready_eat_30day_bi = case_when(
      num_ready_eat_30day >= mean(model_df$num_ready_eat_30day, na.rm = TRUE) + (3 * sd(model_df$num_ready_eat_30day, na.rm = TRUE)) ~ 1,
     num_ready_eat_30day < mean(model_df$num_ready_eat_30day, na.rm = TRUE) + (3 * sd(model_df$num_ready_eat_30day, na.rm = TRUE)) ~ 0,
     TRUE ~ NA_integer_
    ),
    num_frozen_meal_30day_bi = case_when(
      num_frozen_meal_30day >= mean(model_df$num_frozen_meal_30day, na.rm = TRUE) + (3 * sd(model_df$num_frozen_meal_30day, na.rm = TRUE)) ~ 1,
     num_frozen_meal_30day < mean(model_df$num_frozen_meal_30day, na.rm = TRUE) + (3 * sd(model_df$num_frozen_meal_30day, na.rm = TRUE)) ~ 0,
     TRUE ~ NA_integer_
    ),
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
        dr_told_sleep_trouble,
        dr_told_sleep_disorder,
        smoke_100cig_life
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
      num_ready_eat_30day,
      num_frozen_meal_30day,
      sex
    )
  )

# don't start this yet
pred_matrix <- make.predictorMatrix(data = model_df)
imp_method <- make.method(data = model_df)

# pred_matrix[, "seqn"] <- 0
# pred_matrix[, "birth_country"] <- 0
# pred_matrix[c("told_prediabetes", "told_risk_diabetes", "could_risk_diabetes"), "total_chol_mg_dl"] <- 0

prac <- 
  mice(
  model_df,
  maxit = 0,
  m = 1,
  method = imp_method,
  predictorMatrix = pred_matrix
)

prac
prac$loggedEvents

rm(prac)
gc()

inspectdf::inspect_na(model_df)

set.seed(101125)
model_imp <- 
  mice(
    model_df,
    m = 1,
    maxit = 1,
    method = imp_method,
    predictorMatrix = pred_matrix
  )

model_imp$loggedEvents

# saveRDS(model_imp, here::here("projects/nhanes/imputation_model_more_var.rds"))


plot(model_imp, layout = c(10, 10))
densityplot(model_imp)
bwplot(model_imp)
xyplot(model_imp, trigly_mg_dl ~ total_chol_mg_dl)

propplot <- function(x, formula, facet = "wrap", ...) {
  library(ggplot2)

  cd <- data.frame(mice::complete(x, "long", include = TRUE))
  cd$.imp <- factor(cd$.imp)
  
  r <- as.data.frame(is.na(x$data))
  
  impcat <- x$meth != "" & sapply(x$data, is.factor)
  vnames <- names(impcat)[impcat]
  
  if (missing(formula)) {
    formula <- as.formula(paste(paste(vnames, collapse = "+",
                                      sep = ""), "~1", sep = ""))
  }
  
  tmsx <- terms(formula[-3], data = x$data)
  xnames <- attr(tmsx, "term.labels")
  xnames <- xnames[xnames %in% vnames]
  
  if (paste(formula[3]) != "1") {
    wvars <- gsub("[[:space:]]*\\|[[:print:]]*", "", paste(formula)[3])
    # wvars <- all.vars(as.formula(paste("~", wvars)))
    wvars <- attr(terms(as.formula(paste("~", wvars))), "term.labels")
    if (grepl("\\|", formula[3])) {
      svars <- gsub("[[:print:]]*\\|[[:space:]]*", "", paste(formula)[3])
      svars <- all.vars(as.formula(paste("~", svars)))
    } else {
      svars <- ".imp"
    }
  } else {
    wvars <- NULL
    svars <- ".imp"
  }
  
  for (i in seq_along(xnames)) {
    xvar <- xnames[i]
    select <- cd$.imp != 0 & !r[, xvar]
    cd[select, xvar] <- NA
  }
  
  
  for (i in which(!wvars %in% names(cd))) {
    cd[, wvars[i]] <- with(cd, eval(parse(text = wvars[i])))
  }
  
  meltDF <- reshape2::melt(cd[, c(wvars, svars, xnames)], id.vars = c(wvars, svars))
  meltDF <- meltDF[!is.na(meltDF$value), ]
  
  
  wvars <- if (!is.null(wvars)) paste0("`", wvars, "`")
  
  a <- plyr::ddply(meltDF, c(wvars, svars, "variable", "value"), plyr::summarize,
             count = length(value))
  b <- plyr::ddply(meltDF, c(wvars, svars, "variable"), plyr::summarize,
             tot = length(value))
  mdf <- merge(a,b)
  mdf$prop <- mdf$count / mdf$tot
  
  plotDF <- merge(unique(meltDF), mdf)
  plotDF$value <- factor(plotDF$value,
                         levels = unique(unlist(lapply(x$data[, xnames], levels))),
                         ordered = T)
  
  p <- ggplot(plotDF, aes(x = value, fill = get(svars), y = prop)) +
    geom_bar(position = "dodge", stat = "identity") +
    theme(legend.position = "bottom", ...) +
    ylab("proportion") +
    scale_fill_manual(name = "",
                      values = c("black",
                                 colorRampPalette(
                                   RColorBrewer::brewer.pal(9, "Blues"))(x$m + 3)[1:x$m + 3])) +
    guides(fill = guide_legend(nrow = 1))
  
  if (facet == "wrap")
    if (length(xnames) > 1) {
      print(p + facet_wrap(c("variable", wvars), scales = "free"))
    } else {
      if (is.null(wvars)) {
        print(p)
      } else {
        print(p + facet_wrap(wvars, scales = "free"))
      }
    }
  
  if (facet == "grid")
    if (!is.null(wvars)) {
      print(p + facet_grid(paste(paste(wvars, collapse = "+"), "~ variable"),
                           scales = "free"))
    }
}

propplot(model_imp)