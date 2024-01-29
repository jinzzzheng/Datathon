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
from datetime import datetime
from datetime import date

# read
df = pd.read_parquet('/Users/yujinzheng/Downloads/cs2040/Datathon/data/catB_train.parquet')
original_df = df
hidden_data = df


# write
#df.to_parquet('my_newfile.parquet')

# Investigate target column

# Step 1: Clean f_purchase_lh
df['f_purchase_lh'] = df['f_purchase_lh'].fillna(0)


label_encoder = LabelEncoder()
df['clttype'] = label_encoder.fit_transform(df['clttype'])
df['annual_income_est']= label_encoder.fit_transform(df['annual_income_est'])
df ['cltdob_fix'] = df['cltdob_fix'].replace(to_replace='None', value=np.nan).dropna()
df['cltdob_fix'] = pd.to_datetime(df['cltdob_fix'], format='%Y-%m-%d')
def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
df['age'] = df['cltdob_fix'].apply(calculate_age)
df['age'] = df['age'].dropna()

### CORRELATION MATRIX
columns_to_test = ["flg_gi_claim","flg_is_proposal","is_housewife_retiree", "is_sg_pr", "is_class_1_2","annual_income_est","n_months_last_bought_products"
                   ,"flg_latest_being_lapse","recency_lapse", "recency_cancel","tot_inforce_pols","f_mindef_mha","recency_clmcon","recency_giclaim","age","f_purchase_lh"]
'''
# Test Correlation matrix
corr_matrix = df.corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
'''
### CORRELATION MATRIX

### CLEANING DATA
columns_to_keep = ["flg_gi_claim","flg_is_proposal","is_housewife_retiree", "is_sg_pr", "is_class_1_2","annual_income_est","n_months_last_bought_products"
                   ,"flg_latest_being_lapse","recency_lapse", "recency_cancel","tot_inforce_pols","f_mindef_mha","recency_clmcon","recency_giclaim","age","f_purchase_lh"]

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

X = df[["flg_gi_claim","flg_is_proposal","is_housewife_retiree", "is_sg_pr", "is_class_1_2","annual_income_est","n_months_last_bought_products"
                   ,"flg_latest_being_lapse","recency_lapse", "recency_cancel","tot_inforce_pols","f_mindef_mha","recency_clmcon","recency_giclaim","age"]]
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

'''
# Initialize Gaussian Naive Bayes classifier
clf = GaussianNB()

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)
'''

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
columns_to_keep = ["flg_gi_claim","flg_is_proposal","is_housewife_retiree", "is_sg_pr", "is_class_1_2","annual_income_est","n_months_last_bought_products"
                   ,"flg_latest_being_lapse","recency_lapse", "recency_cancel","tot_inforce_pols","f_mindef_mha","recency_clmcon","recency_giclaim","age"]
hidden_data = hidden_data[columns_to_keep]
hidden_data = hidden_data.fillna(0)
result = svm_classifier.predict(hidden_data)
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
