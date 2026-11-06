# Greg Pologruto
# This R script will be the naive bayes model
# Goal: predict failed_within_4q (bank fails within the next 4 quarters)
# Graphs are only displayed; metrics are saved to results/nb_metrics.csv

# load packages
pacman::p_load(e1071, caret, tidyverse, readr, scales)

# set seed to make results reproducible
set.seed(42)

# load data from github repo
train_data <- readr::read_csv("https://raw.githubusercontent.com/greg-pologruto/Data-Challenge-AG-Team/main/data/bank_model_train_downsampled.csv",
                              show_col_types = FALSE)
test_data  <- readr::read_csv("https://raw.githubusercontent.com/greg-pologruto/Data-Challenge-AG-Team/main/data/bank_model_test.csv",
                              show_col_types = FALSE)


#  1. Separate x and y 
# Same predictors as the xgboost model so the two can be compared fairly

continuous_cols <- c(
  "equity_to_assets_pct", "leverage_ratio", "total_risk_based_capital_ratio",
  "noncurrent_rate_pct", "past_due_30_89_to_assets_pct", "allowance_to_loans_pct",
  "roa_annualized_pct", "nii_to_assets_annualized_pct",
  "cre_concentration_pct", "loan_to_deposit_pct", "uninsured_deposit_pct",
  "brokered_to_assets_pct", "borrowed_to_assets_pct", "cash_to_assets_pct",
  "securities_to_assets_pct", "log_total_assets", "county_unemp_rate"
)
factor_cols <- c("charter_class", "county_disaster_past_4q")

# Naive Bayes models each numeric feature as a bell curve within each class, so the
# extreme outliers (e.g. loan_to_deposit_pct in the millions) would wreck the fitted
# means and SDs. Cap each feature at its 1st/99th percentile, using TRAINING
# percentiles for both sets so no test information leaks in.
caps <- map(continuous_cols, \(v) quantile(train_data[[v]], c(0.01, 0.99),
                                           na.rm = TRUE, names = FALSE)) |>
  set_names(continuous_cols)

prep_nb <- function(df) {
  out <- df |> select(all_of(c(continuous_cols, factor_cols)))
  for (v in continuous_cols) {
    out[[v]] <- pmin(pmax(out[[v]], caps[[v]][1]), caps[[v]][2])
  }
  out |>
    mutate(charter_class           = factor(charter_class, levels = c("N", "SM", "NM")),
           county_disaster_past_4q = factor(county_disaster_past_4q, levels = c(0, 1))) |>
    as.data.frame()
}

train_x <- prep_nb(train_data)
train_y <- factor(train_data$failed_within_4q, levels = c(0, 1))
train_w <- train_data$sample_weight          # used to rescale probabilities

test_x <- prep_nb(test_data)
test_y <- test_data$failed_within_4q


#  2. Create model 
# laplace = 1 smooths the categorical features so no level gets a probability of 0.
nb_model <- naiveBayes(x = train_x, y = train_y, laplace = 1)

cat("Class priors learned from the (downsampled) training data:\n")
print(nb_model$apriori / sum(nb_model$apriori))


#  3. Predict on the test set 
test_prob_raw <- predict(nb_model, test_x, type = "raw")[, "1"]

# The model's prior (~9% failures) comes from the downsampled training data.
# Rescale to the real failure rate, the same way as in the xgboost script.
neg_w     <- unique(train_w[train_data$failed_within_4q == 0])[1]     # ~9.25
test_prob <- test_prob_raw / (test_prob_raw + (1 - test_prob_raw) * neg_w)


#  4. Results 
auc_roc <- function(p, y) {
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
avg_precision <- function(p, y) {          # PR-AUC (average precision)
  y <- y[order(p, decreasing = TRUE)]
  precision <- cumsum(y) / seq_along(y)
  sum(precision[y == 1]) / sum(y)
}

# Flag the riskiest 5% of test bank-quarters (same rule as xgboost).
# Naive Bayes often gives probabilities of exactly 1, so ties can push the
# flagged share slightly above 5%
flag_share <- 0.05
cutoff     <- quantile(test_prob, 1 - flag_share)
test_pred  <- as.integer(test_prob >= cutoff)

cm <- caret::confusionMatrix(factor(test_pred, levels = c(0, 1)),
                             factor(test_y,    levels = c(0, 1)),
                             positive = "1")
print(cm)

metrics <- tibble(
  model              = "naive_bayes",
  test_rows          = length(test_y),
  test_failures      = sum(test_y),
  failure_rate_pct   = 100 * mean(test_y),
  roc_auc            = auc_roc(test_prob, test_y),
  pr_auc             = avg_precision(test_prob, test_y),
  pr_auc_baseline    = mean(test_y),             # what a random guess would score
  flagged_share      = flag_share,
  actual_flagged_share = mean(test_pred),
  recall_flagged     = sum(test_pred == 1 & test_y == 1) / sum(test_y == 1),
  precision_flagged  = sum(test_pred == 1 & test_y == 1) / sum(test_pred == 1)
)
print(metrics, width = Inf)

# This feature importance was not a gain metric like xgb model so I had AI write a way to compare the importance of log-odds in the model
# this is so we can compare across models, not specific numbers but which the rank of variables can be compared

#  5. Feature importance 
# Naive Bayes has no built-in importance score. But its prediction is a SUM of
# per-feature pieces: log-odds = prior + sum over features of
#   log P(x | failed) - log P(x | survived).
# A feature's importance = how far, on average, it moves a test bank's log-odds.
nb_contributions <- function(model, newdata) {
  map_dfc(names(model$tables), \(v) {
    tab <- model$tables[[v]]
    x   <- newdata[[v]]
    contrib <- rep(0, length(x))                 
    ok  <- !is.na(x)
    if (is.numeric(x)) {
      # numeric tables: column 1 = mean, column 2 = SD, one row per class
      sd0 <- max(tab["0", 2], 1e-6); sd1 <- max(tab["1", 2], 1e-6)
      contrib[ok] <- dnorm(x[ok], tab["1", 1], sd1, log = TRUE) -
        dnorm(x[ok], tab["0", 1], sd0, log = TRUE)
    } else {
      lv <- as.character(x[ok])
      contrib[ok] <- log(tab["1", lv]) - log(tab["0", lv])
    }
    tibble(!!v := contrib)
  })
}

contrib <- nb_contributions(nb_model, test_x)

importance <- tibble(
  Feature    = names(contrib),
  Importance = map_dbl(contrib, \(z) mean(abs(z)))
) |>
  arrange(desc(Importance))
print(importance, n = Inf)

p_imp <- importance |>
  slice_max(Importance, n = 15) |>
  mutate(Feature = fct_reorder(Feature, Importance)) |>
  ggplot(aes(Importance, Feature)) +
  geom_col(fill = "#4C78A8") +
  labs(title = "Naive Bayes feature importance (top 15)",
       subtitle = "Average absolute shift each feature gives a test bank's log-odds of failure",
       x = "Mean |contribution to log-odds|", y = NULL) +
  theme_minimal()
print(p_imp)


#  6. Confusion matrix plot
cm_df <- as.data.frame(cm$table) |>                 # columns: Prediction, Reference, Freq
  mutate(
    Prediction = factor(Prediction, levels = c(0, 1), labels = c("Not flagged", "Flagged")),
    Reference  = factor(Reference,  levels = c(0, 1), labels = c("Survived", "Failed within 4q")),
    correct    = (Prediction == "Flagged") == (Reference == "Failed within 4q")
  ) |>
  group_by(Reference) |>
  mutate(pct_of_actual = Freq / sum(Freq)) |>       # share of each actual class
  ungroup()

p_cm <- ggplot(cm_df, aes(Reference, Prediction, fill = correct)) +
  geom_tile(colour = "white", linewidth = 1.5, alpha = 0.85) +
  geom_text(aes(label = sprintf("%s\n(%.1f%% of actual)",
                                scales::comma(Freq), 100 * pct_of_actual)),
            size = 5, lineheight = 0.9) +
  scale_fill_manual(values = c(`TRUE` = "#9ECAE1", `FALSE` = "#F4A6A6"),
                    labels = c(`TRUE` = "Correct", `FALSE` = "Error"), name = NULL) +
  scale_y_discrete(limits = rev) +                  # "Flagged" on top
  labs(title = "Naive Bayes confusion matrix (test set, 2011-2012)",
       subtitle = sprintf("Riskiest %.0f%% of bank-quarters flagged | Recall = %.1f%% | Precision = %.1f%%",
                          100 * metrics$actual_flagged_share,
                          100 * metrics$recall_flagged,
                          100 * metrics$precision_flagged),
       x = "Actual outcome", y = "Model prediction") +
  theme_minimal(base_size = 13) +
  theme(panel.grid = element_blank(), legend.position = "bottom")
print(p_cm)


#  7. Save metrics into csv -
write_csv(metrics, "results/naive_bayes_metrics.csv")

# commented out, this code saves the files as png, matches the format of the plots from xgboost

# png("results/naive_bayes_feature_importance.png", width = 2100, height = 1800, res = 300, bg = "white")
# print(p_imp)
# dev.off()
# 
# png("results/naive_bayes_confusion_matrix.png", width = 2100, height = 1800, res = 300, bg = "white")
# print(p_cm)
# dev.off()
