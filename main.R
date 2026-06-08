library(xgboost)
library(dplyr)

# read in training data
df = read.csv('data/train.csv')

# explore data
str(df)

summary(df)

hist(df$u)

table(df$galaxy_population)

# training variables and target variable
X = df %>%
  select(-id, -spectral_type, -galaxy_population, -class)
y = df$class

mappings = levels(as.factor(y))
mappings

y = as.numeric(as.factor(y)) - 1

# splitting train/test
train_test_split = sample(c(TRUE, FALSE), nrow(df), replace = TRUE, prob = c(0.8, 0.2))

head(train_test_split)

X_train = X[train_test_split,]
X_test = X[!train_test_split,]

y_train = y[train_test_split]
y_test = y[!train_test_split]

# train model

model = xgboost(
  data = as.matrix(X_train),
  label = y_train,
  objective = "multi:softmax",
  num_class = 3,
  nrounds = 100
)

# make predictions
predictions = predict(model, as.matrix(X_test))
table(predictions)

df_check = data.frame(
  "Actual" = y_test,
  "Predicted" = predictions
)

length(which(df_check$Actual == df_check$Predicted)) / nrow(df_check) * 100

# prepare competition predictions
df_test = read.csv('data/test.csv')

str(df_test)

df_test_f = df_test %>%
  select(-id, -spectral_type, -galaxy_population)
str(df_test_f)

comp_preds = predict(model, as.matrix(df_test_f))
table(comp_preds)

# create submission csv
comp_submission_df = data.frame(
  "id" = df_test$id,
  "class" = comp_preds
) %>%
  mutate(
    class = case_when(
      class == 0 ~ mappings[1],
      class == 1 ~ mappings[2],
      class == 2 ~ mappings[3]
    )
  )

write.csv(comp_submission_df, "comp_submission_R.csv", row.names = FALSE)
