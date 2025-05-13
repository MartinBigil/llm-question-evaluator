# Load necessary libraries
library(tidyverse)
library(gt)
library(stringr)
library(scales)
library(emmeans)
library(lme4)
library(scales)

# Read dataset
df <- read_csv("combined_llm_results_cleaned (5).csv")

# === CLEAN & SUMMARIZE ===

# Accuracy by model
accuracy_by_model <- df %>%
  group_by(model) %>%
  summarise(overall_accuracy = mean(is_correct, na.rm = TRUE))

# Agreement rate (clean followup_pred)
agreement_by_model <- df %>%
  filter(category == "ethical") %>%
  mutate(
    followup_pred_clean = str_trim(str_to_lower(followup_pred)),
    followup_agreed = followup_pred_clean == "yes"
  ) %>%
  group_by(model) %>%
  summarise(agreement_rate = mean(followup_agreed, na.rm = TRUE))

# Merge summary
summary_table <- accuracy_by_model %>%
  left_join(agreement_by_model, by = "model")

# === NICE SUMMARY TABLE ===

summary_table %>%
  gt() %>%
  fmt_percent(columns = c(overall_accuracy, agreement_rate), decimals = 1) %>%
  cols_label(
    model = "LLM Model",
    overall_accuracy = "Overall Accuracy",
    agreement_rate = "Follow-up Agreement Rate"
  ) %>%
  tab_header(
    title = "Model Accuracy and Ethical Agreement Rates"
  ) %>%
  tab_options(
    table.font.names = "sans",
    heading.title.font.size = 14,
    table.font.size = 12,
    data_row.padding = px(5)
  )

# === FIGURES ===


# Compute accuracy by model and source
accuracy_model_source <- df %>%
  group_by(model, source) %>%
  summarise(accuracy = mean(is_correct, na.rm = TRUE), .groups = "drop") %>%
  mutate(label = percent(accuracy, accuracy = 1))

# Plot grouped bar chart with percent labels
ggplot(accuracy_model_source, aes(x = model, y = accuracy, fill = source)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.65, color = "black") +
  geom_text(
    aes(label = label),
    position = position_dodge(width = 0.7),
    vjust = -0.4,
    size = 3.5,
    family = "sans"
  ) +
  scale_fill_brewer(palette = "Set2", name = "Question Source") +
  labs(
    title = "Model Accuracy by Question Source",
    x = "Model",
    y = "Accuracy"
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.1)) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 30, hjust = 1)
  )

# Recalculate grouped accuracy
df <- df %>%
  mutate(group = paste(model, source, sep = "_"))

accuracy_grouped <- df %>%
  group_by(group, category) %>%
  summarise(accuracy = mean(is_correct, na.rm = TRUE), .groups = "drop") %>%
  mutate(label = percent(accuracy, accuracy = 1))

# Plot with % labels on top
ggplot(accuracy_grouped, aes(x = group, y = accuracy, fill = category)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, color = "black") +
  geom_text(
    aes(label = label),
    position = position_dodge(width = 0.7),
    vjust = -0.5,
    size = 2,
    family = "sans"
  ) +
  scale_fill_brewer(palette = "Dark2", name = "Category") +
  labs(
    title = "Accuracy by Model, Source, and Category",
    x = NULL,
    y = "Accuracy"
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.1)) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 40, hjust = 1)
  )

##########################################################################

STATISTICAL ANALYSIS PORTION WIT LOGISTIC REGRESSIONS, CHIQSQ, ANOVA ETC...

##########################################################################
# Convert predictors to factors
df <- df %>%
  mutate(
    model = as.factor(model),
    source = as.factor(source),
    category = as.factor(category)
  )

# Logistic regression: does accuracy depend on model, source, category?
logit_model <- glm(is_correct ~ model + source + category, data = df, family = binomial())

summary(logit_model)

chisq_table <- table(df$model, df$is_correct)
chisq.test(chisq_table)

# Filter to ethical questions
df_ethical <- df %>% filter(category == "ethical")

# Fit logistic model: accuracy ~ model + source
logit_ethical <- glm(is_correct ~ model + source, data = df_ethical, family = binomial())
summary(logit_ethical)

# Full interaction model
logit_interact <- glm(is_correct ~ model * category + source, data = df, family = binomial())
summary(logit_interact)



# Fit mixed model: random intercept for question
logit_mixed <- glmer(is_correct ~ model + source + category + (1 | question_id), 
                     data = df, family = binomial())

summary(logit_mixed)


# Base diagnostic plots for glm
par(mfrow = c(2, 2))  # 2x2 layout
plot(logit_model)

AIC(logit_model, logit_interact, logit_mixed)

# Quick check for overdispersion
overdispersion_stat <- sum(residuals(logit_model, type = "pearson")^2) / logit_model$df.residual
print(paste("Overdispersion ratio:", round(overdispersion_stat, 2)))


# Generate marginal predicted probabilities by model and category
emm_results <- emmeans(logit_mixed, ~ model * category, type = "response") %>%
  as.data.frame()

# View the results
print(emm_results)

# Optional: Save a simplified table for use in LaTeX
write.csv(emm_results, "marginal_predicted_probs.csv", row.names = FALSE)

# Plot with 95% CIs
ggplot(emm_results, aes(x = model, y = prob, fill = category)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_errorbar(
    aes(ymin = asymp.LCL, ymax = asymp.UCL),
    position = position_dodge(width = 0.7),
    width = 0.2
  ) +
  labs(
    title = "Marginal Predicted Accuracy by Model and Category",
    x = "LLM Model",
    y = "Predicted Probability (Accuracy)"
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
  theme_minimal(base_family = "sans") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

