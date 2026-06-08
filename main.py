from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost

# read in data and explore
df = pd.read_csv("data/train.csv")

df.describe
df.columns

# Create training vars and test labels
X = df.drop(labels = ["id","spectral_type","galaxy_population","class"], axis=1)
X.columns

y = df["class"]

# y needs to be a factor
y = pd.factorize(df["class"])[0]
mappings = pd.factorize(df["class"])[1]

# split test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

# train model
model = xgboost.XGBClassifier(n_estimators = 100, random_state=123)
model.fit(X_train, y_train)

# predictions on test set
preds = model.predict(X_test)
preds

# check preds
check_df = pd.DataFrame({
    "Actual" : y_test,
    "Predicted" : preds
})

sum(check_df["Actual"] == check_df["Predicted"]) / len(check_df) * 100

# predict on competition submission data
df_comp = pd.read_csv("data/test.csv")
df_comp.columns

df_comp_f = df_comp.drop(labels=["id","spectral_type","galaxy_population"], axis=1)

comp_preds = model.predict(df_comp_f)

comp_submission_df = pd.DataFrame({
    "id" : df_comp["id"],
    "class" : comp_preds
})

comp_submission_df["class"] = comp_submission_df["class"].map({0: mappings[0], 1: mappings[1], 2: mappings[2]})

comp_submission_df.to_csv("comp_submission_Python.csv", index=False)
