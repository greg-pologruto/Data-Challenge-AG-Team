# Greg Pologruto
# Comparison table: XGBoost vs Naive Bayes (test set, 2011-2012)
# Reads the two metrics csvs, derives the confusion-matrix metrics, and draws a
# styled table with the best model (highest PR AUC) highlighted.

pacman::p_load(tidyverse, readr)

#  1. Load the metrics
xgb <- read_csv("results/xgb_metrics.csv",         show_col_types = FALSE) |> mutate(model = "XGBoost")
nb  <- read_csv("results/naive_bayes_metrics.csv", show_col_types = FALSE) |> mutate(model = "Naive Bayes")

keep <- c("model", "test_rows", "test_failures", "roc_auc", "pr_auc",
          "recall_flagged", "precision_flagged")


#  2. Rebuild the confusion matrix and compute the metrics 
# The csvs store recall and precision for the "flag the riskiest 5%" rule.
# From those plus the test-set counts we can recover TP / FP / TN / FN exactly.
comparison <- bind_rows(select(xgb, all_of(keep)), select(nb, all_of(keep))) |>
  mutate(
    P       = test_failures,                 # actual failures
    N       = test_rows - test_failures,     # actual survivors
    TP      = round(recall_flagged * P),
    flagged = round(TP / precision_flagged), # TP + FP
    FP      = flagged - TP,
    TN      = N - FP,
    FN      = P - TP,
    Accuracy    = (TP + TN) / test_rows,
    Sensitivity = TP / P,                    # = recall
    Specificity = TN / N,
    Precision   = TP / flagged,
    F1          = 2 * Precision * Sensitivity / (Precision + Sensitivity),
    `ROC AUC`   = roc_auc,
    `PR AUC`    = pr_auc
  )

# Counts, for checking against the confusion matrix plots
print(comparison |> select(model, TP, FP, TN, FN))

comparison <- comparison |>
  select(Model = model, Accuracy, Sensitivity, Specificity, Precision, F1,
         `ROC AUC`, `PR AUC`) |>
  arrange(desc(`PR AUC`))
print(comparison, width = Inf)

best_model <- comparison$Model[1]   # highest PR AUC = the highlighted row


#  3. Build the table graphic 
tbl_chr   <- comparison |> mutate(across(-Model, ~ sprintf("%.3f", .x)))
col_names <- names(tbl_chr)
n_rows    <- nrow(tbl_chr)

# column positions (Model column wider than the number columns)
widths <- c(2.4, rep(1.45, length(col_names) - 1))
x_max  <- cumsum(widths)
x_min  <- x_max - widths

row_type <- c("header", if_else(tbl_chr$Model == best_model, "best", "other"))

# one cell (box + text label) per column per row; row 0 = header
cells_df <- bind_rows(
  tibble(row = 0, column = col_names, label = col_names),
  tbl_chr |>
    mutate(row = row_number()) |>
    pivot_longer(-row, names_to = "column", values_to = "label")
) |>
  mutate(
    col   = match(column, col_names),
    type  = row_type[row + 1],
    xmin  = x_min[col],
    xmax  = x_max[col],
    x     = if_else(col == 1, xmin + 0.2, (xmin + xmax) / 2),
    hjust = if_else(col == 1, 0, 0.5),
    face  = if_else(type %in% c("header", "best"), "bold", "plain")
  )

# plot
p_table <- ggplot(cells_df) +
  geom_rect(aes(xmin = xmin, xmax = xmax, ymin = -row - 0.5, ymax = -row + 0.5, fill = type),
            colour = "grey45", linewidth = 0.6, alpha = 0.85) +   
  geom_text(aes(x = x, y = -row, label = label, hjust = hjust, fontface = face),
            size = 6, colour = "black") +
  scale_fill_manual(values = c(header = "#9ECAE1", best = "#A1D99B", other = "white"),
                    guide = "none") +
  coord_cartesian(expand = FALSE, clip = "off") +
  labs(caption = paste0(
    "Test set: 2011-2012 bank-quarters (320 failures). Threshold metrics flag the riskiest 5% of bank-quarters.\n",
    "Highlighted: ", best_model, " (highest PR AUC)."
  )) +
  theme_void() +
  theme(plot.caption  = element_text(hjust = 0, size = 11, colour = "grey30",
                                     margin = margin(t = 8)),
        plot.margin   = margin(10, 10, 10, 10),
        plot.background = element_rect(fill = "white", colour = NA))

print(p_table)

# Optional: save it (12 x 3 in at 300 dpi = 3600 x 900 px; wider than the other
# plots because the table has 8 columns)
# ggsave("results/model_comparison_table.png", p_table,
#        width = 12, height = 3, units = "in", dpi = 300, bg = "white")
