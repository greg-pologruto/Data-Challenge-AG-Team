# Greg Pologruto
# This R script will be the xgboost model
# Goal: predict failed_within_4q (bank fails within the next 4 quarters)

# load packages
pacman::p_load(xgboost, caret, tidyverse, readr)

# set seed to make results reproducible
set.seed(42)

# load data from github repo
train_data <- readr::read_csv("https://raw.githubusercontent.com/greg-pologruto/Data-Challenge-AG-Team/main/data/bank_model_train_downsampled.csv",
                              show_col_types = FALSE)
test_data  <- readr::read_csv("https://raw.githubusercontent.com/greg-pologruto/Data-Challenge-AG-Team/main/data/bank_model_test.csv",
                              show_col_types = FALSE)


# 1. Separate x and y 
# The test file has 28 columns (no sample_weight), so columns are picked by NAME.

# ai generated list of variables to be dropped
drop_cols <- c(
  "cert", "report_date", "year", "quarter",   # IDs / time, not predictors
  "split", "sample_weight",                   # bookkeeping columns
  "failed_within_4q",                         # the y variable
  "hq_state",                                 # 50+ categories; leave out of the model
  "fed_funds", "treasury_10y"                 # one value per quarter for every bank (a time proxy);
  # test-period rates sit below the training range
)

# xgboost needs an all-numeric matrix, so charter_class becomes 0/1 columns
# (NM is the baseline). Missing values can stay as NA: xgboost handles them.
prep_x <- function(df) {
  df |>
    mutate(charter_N  = as.integer(charter_class == "N"),
           charter_SM = as.integer(charter_class == "SM")) |>
    select(-charter_class, -any_of(drop_cols)) |>
    as.matrix()
}

train_x <- prep_x(train_data)
train_y <- train_data$failed_within_4q
train_w <- train_data$sample_weight          # only used to rescale probabilities

test_x <- prep_x(test_data)
test_y <- test_data$failed_within_4q


cat("Predictors used:", ncol(train_x), "\n"); print(colnames(train_x))


# 2. Validation split for choosing the number of trees
# Train on 2007-2009, validate on 2010, so tuning mimics predicting a later period.
is_val <- train_data$year == 2010

dtrain_fit <- xgb.DMatrix(train_x[!is_val, ], label = train_y[!is_val])
dval       <- xgb.DMatrix(train_x[ is_val, ], label = train_y[ is_val])


# 3. Create model 
params <- list(
  objective        = "binary:logistic",  # outputs a probability of failure
  eval_metric      = "aucpr",            # precision-recall AUC: suited to rare events
  eta              = 0.05,               # learning rate (smaller = slower but steadier)
  max_depth        = 4,                  # depth of each tree
  min_child_weight = 5,                  # minimum size of a leaf (guards against overfitting)
  subsample        = 0.8,                # share of rows used per tree
  colsample_bytree = 0.8,                # share of columns used per tree
  seed             = 42
)

# Find the best number of trees with early stopping on the 2010 validation set
val_model <- xgb.train(
  params                = params,
  data                  = dtrain_fit,
  nrounds               = 2000,
  evals                 = list(train = dtrain_fit, val = dval),
  early_stopping_rounds = 50,     # stop when validation PR-AUC hasn't improved for 50 rounds
  print_every_n         = 50,
  verbose               = 1
)

# best_nrounds was returning null so these checks are to make sure we pick a best amount of trees
# used AI to write this method and find the correct number of trees to use
best_nrounds <- attr(val_model, "best_iteration")

if (is.null(best_nrounds)) {
  # some versions store it only inside the booster, 0-based and as text
  bi <- xgb.attr(val_model, "best_iteration")
  if (!is.null(bi)) best_nrounds <- as.integer(bi) + 1
}

if (is.null(best_nrounds)) {
  # last resort: read it from the validation scores logged each round
  eval_log <- attr(val_model, "evaluation_log")
  best_nrounds <- which.max(eval_log$val_aucpr)
}

stopifnot(!is.null(best_nrounds), length(best_nrounds) == 1)
cat("Best number of trees:", best_nrounds, "\n")   # for this seed the best number of trees is 41 ***

# ------

# Refit on ALL training data (2007-2010) with that number of trees
dtrain_all  <- xgb.DMatrix(train_x, label = train_y)
final_model <- xgb.train(params = params, data = dtrain_all,
                         nrounds = best_nrounds, verbose = 0)


# ---- 4. Results ---------------------------------------------------------------
dtest         <- xgb.DMatrix(test_x)
test_prob_raw <- predict(final_model, dtest)

# Healthy banks were downsampled in training, so raw probabilities are inflated
# (about 9x). Rescale them to the real failure rate using the healthy-row weight.
neg_w     <- unique(train_w[train_y == 0])[1]     # ~9.25
test_prob <- test_prob_raw / (test_prob_raw + (1 - test_prob_raw) * neg_w)

# Metric helpers
auc_roc <- function(p, y) {
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
avg_precision <- function(p, y) {          # PR-AUC (average precision)
  y <- y[order(p, decreasing = TRUE)]
  precision <- cumsum(y) / seq_along(y)
  sum(precision[y == 1]) / sum(y)
}

# Flag the riskiest 5% of test bank-quarters and see how many failures are caught
flag_share <- 0.05
cutoff     <- quantile(test_prob, 1 - flag_share)
test_pred  <- as.integer(test_prob >= cutoff)

cm <- caret::confusionMatrix(factor(test_pred, levels = c(0, 1)),
                             factor(test_y,    levels = c(0, 1)),
                             positive = "1")
print(cm)

metrics <- tibble(
  test_rows          = length(test_y),
  test_failures      = sum(test_y),
  failure_rate_pct   = 100 * mean(test_y),
  roc_auc            = auc_roc(test_prob, test_y),
  pr_auc             = avg_precision(test_prob, test_y),
  pr_auc_baseline    = mean(test_y),             # what a random guess would score
  flagged_share      = flag_share,
  recall_flagged     = sum(test_pred == 1 & test_y == 1) / sum(test_y == 1),
  precision_flagged  = sum(test_pred == 1 & test_y == 1) / sum(test_pred == 1),
  best_nrounds       = best_nrounds
)
print(metrics, width = Inf)

# Feature importance (Gain = how much each feature improved the trees)
importance <- xgb.importance(model = final_model) |> as_tibble()
print(importance, n = Inf)

p_imp <- importance |>
  slice_max(Gain, n = 15) |>
  mutate(Feature = fct_reorder(Feature, Gain)) |>
  ggplot(aes(Gain, Feature)) +
  geom_col(fill = "#4C78A8") +
  labs(title = "XGBoost feature importance (top 15)", x = "Gain", y = NULL) +
  theme_minimal()
print(p_imp)


# 5. Save results into csv 
dir.create("results", showWarnings = FALSE)

predictions <- test_data |>
  select(cert, report_date, year, quarter, failed_within_4q) |>
  mutate(pred_prob = test_prob, flagged = test_pred)

write_csv(predictions, "results/xgb_test_predictions.csv")
write_csv(metrics,     "results/xgb_metrics.csv")
write_csv(importance,  "results/xgb_feature_importance.csv")
ggsave("results/xgb_feature_importance.png", p_imp, width = 8, height = 6, bg = "white")


# ---- Confusion matrix plot ----
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
  labs(title = "XGBoost confusion matrix (test set, 2011-2012)",
       subtitle = sprintf("Riskiest %.0f%% of bank-quarters flagged | Recall = %.1f%% | Precision = %.1f%%",
                          100 * flag_share,
                          100 * metrics$recall_flagged,
                          100 * metrics$precision_flagged),
       x = "Actual outcome", y = "Model prediction") +
  theme_minimal(base_size = 13) +
  theme(panel.grid = element_blank(), legend.position = "bottom")

print(p_cm)
ggsave("results/xgb_confusion_matrix.png", p_cm, width = 7, height = 6, bg = "white")
