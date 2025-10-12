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



# don't start this yet
pred_matrix <- make.predictorMatrix(data = latino)
imp_method <- make.method(data = latino)

pred_matrix[, "seqn"] <- 0

pred_matrix[, "birth_country"] <- 0
pred_matrix[c("told_prediabetes", "told_risk_diabetes", "could_risk_diabetes"), "total_chol_mg_dl"] <- 0

imp_method[c(
  "hdl_chol_mg_dl",
  "trigly_mg_dl",
  "ldl_chol_mg_dl",
  "total_chol_mg_dl"
)] <- "midastouch"

imp_method[c(
  "albumin_g_dl",
  "alp_iu_l",
  "ast_u_l",
  "alt_u_l",
  "ggt_u_l",
  "total_bilirubin_mg_dl",
  "bmi",
  "waist_circumference",
  "min_sedentary",
  "num_meals_not_home_prepare",
  "num_ready_eat_food_30day",
  "num_frozen_meal_30day"
)] <- "pmm"

imp_method[c(
  "citizen",
  "length_us",
  "ed",
  "annual_house_income",
  "alc_drink12_yr",
  "ever_45_drink_everyday",
  "gen_health",
  "covered_insurance",
  "told_angina",
  "told_heart_attack",
  "told_liver_cond",
  "told_cancer",
  "dr_told_exercise",          
  "you_control_wt",
  "you_reduce_fat",
  "told_prediabetes",          
  "told_risk_diabetes",
  "could_risk_diabetes",
  "vig_rec_pa",
  "mod_rec_pa",
  "told_hep_b",
  "told_hep_c",
  "told_high_bp",
  "dr_told_high_chol",        
  "hep_a_anti",
  "hep_b_core_anti"
)] <- "cart"

prac <- 
  mice(
  latino,
  maxit = 0,
  m = 1,
  method = imp_method,
  predictorMatrix = pred_matrix
)

prac
prac$loggedEvents

rm(prac)
gc()

set.seed(12345)
model_imp <- 
  futuremice(
    latino,
    m = 60,
    maxit = 30,
    method = imp_method,
    predictorMatrix = pred_matrix,
    parallelseed = 12345,
    n.core = 4
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