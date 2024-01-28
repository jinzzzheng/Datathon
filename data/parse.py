import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import f1_score


# read
df = pd.read_parquet('catB_train.parquet')
original_df = df
hidden_data = df


# write
#df.to_parquet('my_newfile.parquet')

# Investigate target column

# Step 1: Clean f_purchase_lh
df['f_purchase_lh'] = df['f_purchase_lh'].fillna(0)


label_encoder = LabelEncoder()
df['clttype'] = label_encoder.fit_transform(df['clttype'])

### CORRELATION MATRIX
columns_to_test = ["clttype", "stat_flag", "flg_substandard", "flg_has_health_claim", "flg_is_borderline_standard"
                   , "flg_is_revised_term", "flg_is_rental_flat", "flg_has_life_claim", "flg_gi_claim", "flg_is_proposal"
                   , "flg_with_preauthorisation", "is_housewife_retiree", "is_sg_pr", "is_class_1_2"
                   , "is_dependent_in_at_least_1_policy", "f_purchase_lh"]
'''
# Test Correlation matrix
corr_matrix = df.corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
'''
### CORRELATION MATRIX

### CLEANING DATA
columns_to_keep = ["clttype", "flg_gi_claim", "flg_is_proposal", "is_housewife_retiree", "is_sg_pr", "is_class_1_2", "f_purchase_lh"]

df = df[columns_to_keep]
df = df.fillna(0)
print(df.shape)
print(df["f_purchase_lh"].value_counts())

# Separate majority and minority classes
df_majority = df[df['f_purchase_lh'] == 0]
df_minority = df[df['f_purchase_lh'] == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # Sample with replacement
                                 n_samples=17282,  # to match majority class
                                 random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df = df_upsampled
### CLEANING DATA

X = df[["clttype", "flg_gi_claim", "flg_is_proposal", "is_housewife_retiree", "is_sg_pr", "is_class_1_2"]]
y = df["f_purchase_lh"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''
# Initialize the Logistic Regression classifier
logistic_regression = LogisticRegression()

# Train the classifier
logistic_regression.fit(X_train, y_train)
'''


'''
# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

'''

# Initialize Gaussian Naive Bayes classifier
clf = GaussianNB()

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

## Exploratory Data Analysis
#summary_stats = df.describe() # Summary

# Test Univariate feature selection (Cannot cuz not categorical)

# Test Chi Squared

# Make predictions on the test set
# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Training Accuracy:", accuracy)
f1 = f1_score(y_test, y_pred)
print("F1 score:", f1)

count_0 = 0
count_1 = 0
for entry in y_pred:
    if entry == 0:
        count_0 += 1
    if entry == 1:
        count_1 += 1
print(count_0, count_1)


### TEST DATA
print(hidden_data["f_purchase_lh"].value_counts())
hidden_data = hidden_data.drop(columns=["f_purchase_lh"])
hidden_data['clttype'] = label_encoder.fit_transform(hidden_data['clttype']) 
columns_to_keep = ["clttype", "flg_gi_claim", "flg_is_proposal", "is_housewife_retiree", "is_sg_pr", "is_class_1_2"]

hidden_data = hidden_data[columns_to_keep]
hidden_data = hidden_data.fillna(0)
result = clf.predict(hidden_data)
count_0 = 0
count_1 = 0
for entry in result:
    if entry == 0:
        count_0 += 1
    if entry == 1:
        count_1 += 1
print(count_0, count_1)

y_test_hidden = original_df["f_purchase_lh"]
hidden_f1_score = f1_score(y_test_hidden, result)
print("Hidden f1 score:", hidden_f1_score)
