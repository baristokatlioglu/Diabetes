import numpy as np
import pandas as pd
import seaborn as sns
#!pip install missingno
import missingno as msno
from matplotlib import pyplot as plt
from helpers import helpers
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

####
# Exploratory Data Analysis
####

# Reading Data
def load_data():
    df = pd.read_csv("datasets/diabetes.csv")
    return df
df = load_data()
# Review the General Picture
helpers.check_df(df)
# There are 768 observations and 9 variables in the data set.
# The types of the variables appear to be Numeric. (int64, float64)
# There is no NA value in the data set.
# When the quarters were analyzed, the value of 0.95 and above in the Pregnancies Variable caught my attention.
# The minimum value of the Glucose, BloodPressure, SkinThickness, Insulin and BMI variables should be considered to be 0.


# Capture numeric and categorical variables.
for col in df.columns:
    print(f"{col} \n {df[col].nunique()}")
# I looked at the unique class numbers of the variables to determine the threshold values for the categorical and cardinal variables.
# I do not change the default value of grab_col_names 10 and 20.

cat_cols, num_cols, cat_but_car = helpers.grab_col_names(df)
# Of the 9 variables, 1 is categorical and 8 is numerical. The categorical variable is Numeric but Categorical.
# It consists of 1-0s.

# Analyze numerical and categorical variables.

# Analysis of Numerical Variables
helpers.num_summary(df,num_cols)

# Analysis of Categorical Variables
for col in cat_cols:
    helpers.cat_summary(df,col)
# Non-Diabetes make up 65% of the dataset with 500 observations.
# Those with diabetes constitute 34% of the dataset with 268 observations.

# Perform target variable analysis.
helpers.target_summary_with_cat(df, "Outcome", cat_cols)
# Since the target variable was our Single Categorical variable, the averages did not come.

for col in num_cols:
    helpers.target_summary_with_num(df, "Outcome", col)
# The difference in the mean of the glucose variable can cause people to have diabetes.
# The difference in the mean of the insulin variable can cause people to have diabetes.

# Outlier Analysis
for col in num_cols:
    sns.boxplot(df[col])
    plt.show()
# Outlier values were observed in all variables with boxplot. Since it takes q1 = 0.25 q3 = 0.75
# Outliers are detected in all variables.

helpers.num_summary(df,num_cols)
# When we look at the maximum values of the variables, the Pregnancy Number, Skin Thickness and Insulin values are striking.
# I will apply the suppression method with IQR to these variables.

for col in ["Pregnancies", "SkinThickness", "Insulin"]:
    helpers.replace_with_thresholds(df,col)

for col in num_cols:
    print(col, helpers.check_outlier(df,col))

# Perform a Missing Observation Analysis.
df.isnull().sum()
# There are no missing observations in the data set.


# Perform correlation analysis.
helpers.high_correlated_cols(df, True)
# Glucose 0.47, Pregnancies, 0.22, BMI 0.29, Age 0.24 were associated with the target variable.


#####
# Feature Engineering
#####

# Take necessary actions for missing and outlier values.

helpers.num_summary(df,num_cols)
# Glucose, BloodPressure, SkinThickness, Insulin, BMI, Variables cannot have 0 values. I will replace these values with NA

for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    df[col].replace(0, np.nan, inplace=True)

dff = df.copy() # I create dff to test the model that is installed without missing value analysis in the model setup.

# Examining the structure of missing values
msno.matrix(df)
plt.show()
# Deficiencies in the SkinThickness and Insulin variable may be linked. While a missing value is observed in one,
# it is also observed in the other.

# Correlation of missing values
msno.heatmap(df)
plt.show()

na_columns = helpers.missing_values_table(df, True)
helpers.missing_vs_target(df, "Outcome", na_columns)
# All of the missing values in insulin are people with diabetes.
# Missing values in SkinThickness are all people with diabetes.
# When viewed, all missing values are people with diabetes.
# I will fill these values with the Median. I'll try the model with both the padding method and deleting the NA values.

df_rplc = df.copy()
for col in na_columns:
    df_rplc[col].replace(np.nan, df[col].median(), inplace=True)
df_rplc.isnull().sum()
df_rplc.head()
df.dropna(inplace=True)
# !!! I will do the next steps for both df and df_rplc.

# Create new variables.
df["BMI_CAT"] = pd.cut(df["BMI"], bins=[0,18.5,25,30,df.BMI.max()], labels=["Underweight", "Ideal", "Overweight", "Obese"])
df_rplc["BMI_CAT"] = pd.cut(df_rplc["BMI"], bins=[0,18.5,25,30,df_rplc.BMI.max()], labels=["Underweight", "Ideal", "Overweight", "Obese"])
dff["BMI_CAT"] = pd.cut(dff["BMI"], bins=[0,18.5,25,30,dff.BMI.max()], labels=["Underweight", "Ideal", "Overweight", "Obese"])
# I created Weight class from Body Mass indexes.

df["Pregnancy"] = np.where(df["Pregnancies"] > 0, 1, 0)
df_rplc["Pregnancy"] = np.where(df_rplc["Pregnancies"] > 0, 1, 0)
dff["Pregnancy"] = np.where(dff["Pregnancies"] > 0, 1, 0)
# I assigned a value of 1 to people who have experienced pregnancy and 0 to people who have not.

df["Age_CAT"] = pd.cut(df["Age"], bins=(19,25,45,65,df["Age"].max()), labels=["Young Adult", "Adult", "Middle Aged", "Aged"])
df_rplc["Age_CAT"] = pd.cut(df_rplc["Age"], bins=(19,25,45,65,df_rplc["Age"].max()), labels=["Young Adult", "Adult", "Middle Aged", "Aged"])
dff["Age_CAT"] = pd.cut(dff["Age"], bins=(19,25,45,65,dff["Age"].max()), labels=["Young Adult", "Adult", "Middle Aged", "Aged"])
# I divided the people into groups according to their age.

# Perform the Encoding Operations.
df = pd.get_dummies(df, ["Age_CAT", "BMI_CAT"], drop_first=True)
df_rplc = pd.get_dummies(df_rplc, ["Age_CAT", "BMI_CAT"], drop_first=True)
dff = pd.get_dummies(dff, ["Age_CAT", "BMI_CAT"], drop_first=True)
df.drop(["Age", "Pregnancies", "BMI"], axis=1, inplace=True)
df_rplc.drop(["Age", "Pregnancies", "BMI"], axis=1, inplace=True)
dff.drop(["Age", "Pregnancies", "BMI"], axis=1, inplace=True)

# Since categorical Age and BMI variables are not ordinal variables, I encoded with one hot encoder.
# I deleted the encoding variables from the dataset.
# In this section I call grab_col_names again to specify categorical and numeric variables.

cat_cols, num_cols, cat_but_car = helpers.grab_col_names(df)

# Standardize for numeric variables.
rs = RobustScaler()
for col in num_cols:
    df[col+"_robust_scaler"] = rs.fit_transform(df[[col]])
df.head()
for col in num_cols:
    df_rplc[col+"_robust_scaler"] = rs.fit_transform(df_rplc[[col]])
df_rplc.head()
for col in num_cols:
    dff[col+"_robust_scaler"] = rs.fit_transform(dff[[col]])
dff.head()

# Create a Model.
# Model of df with missing values deleted
y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Eksik Değerler Silinen Modelin Accuracy Score'u :", accuracy_score(y_pred, y_test))
# 0.7346938775510204
helpers.plot_importance(rf_model, X_train)
# Glucose .30 , Insulin .16 provides explainability in explaining the target variable.

# Model of df_rplc filled with Median of Missing Values
y_rplc = df_rplc["Outcome"]
X_rplc = df_rplc.drop("Outcome", axis=1)
X_rplc_train, X_rplc_test, y_rplc_train, y_rplc_test = train_test_split(X_rplc, y_rplc, test_size=0.25, random_state=17)
from sklearn.ensemble import RandomForestClassifier
rf_rplc_model = RandomForestClassifier(random_state=46).fit(X_rplc_train, y_rplc_train)
y_rplc_pred = rf_model.predict(X_rplc_test)
print("Eksik Değerlerin Medyan ile doldurulan Modelin Accuracy Score'u :", accuracy_score(y_rplc_pred, y_rplc_test))
# 0.8229166666666666
helpers.plot_importance(rf_model, X_rplc_train)
# Glucose .30, Insulin .16 provides explainability in explaining the target variable.










