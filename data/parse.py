import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score
from datetime import date
from sklearn.ensemble import RandomForestClassifier


### 1. DATA PROCESSING
# read
df = pd.read_parquet('./data/catB_train.parquet')
hidden_data = df.copy(deep=True)

# Investigate target column and understand the data
# Since target column consists only of NaN or 1, convert NaNs to 0.
print(df["f_purchase_lh"].unique())

# Step 1: Clean f_purchase_lh
df['f_purchase_lh'] = df['f_purchase_lh'].fillna(0)

# Label encoder handles non-numerical data such as annual income. Since annual income is in the format: C.60K-100K,
# Convert String labels such as C.60K-100K to labels like 0,1,2,3,4 so that the model can parse the data
label_encoder = LabelEncoder()

## Labelling Client Type
df['clttype'] = label_encoder.fit_transform(df['clttype'])

## Labelling annual income categories
df['annual_income_est']= label_encoder.fit_transform(df['annual_income_est'])

## Convert date of birth to age
df['cltdob_fix'] = df['cltdob_fix'].replace(to_replace='None', value=np.nan).dropna()
df['cltdob_fix'] = pd.to_datetime(df['cltdob_fix'], format='%Y-%m-%d')
def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
df['cltdob_fix'] = df['cltdob_fix'].apply(calculate_age)

## Drop NA for purpose of training the model
df['cltdob_fix'] = df['cltdob_fix'].dropna()
### DATA PROCESSING



### 2. CORRELATION MATRIX & HEATMAP FOR FEATURE SELECTION
## Using Correlation matrix, we selected all features/variables with a correlation of > 5% to the target variable
## Uncomment code below to see Heatmap of correlation matrix
'''
columns_to_test = ["flg_gi_claim","flg_is_proposal","is_housewife_retiree", "is_sg_pr", "is_class_1_2","annual_income_est","n_months_last_bought_products"
                   ,"flg_latest_being_lapse","recency_lapse", "recency_cancel","tot_inforce_pols","f_mindef_mha","recency_clmcon","recency_giclaim","cltdob_fix","f_purchase_lh"]

# Test Correlation matrix
corr_matrix = df.corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
'''
### CORRELATION MATRIX & HEATMAP FOR FEATURE SELECTION


### 3. FILTERING & UPSAMPLING
columns_to_keep = ["flg_gi_claim","flg_is_proposal","is_housewife_retiree", "is_sg_pr", "is_class_1_2","annual_income_est","n_months_last_bought_products"
                   ,"flg_latest_being_lapse","recency_lapse", "recency_cancel","tot_inforce_pols","f_mindef_mha","recency_clmcon","recency_giclaim","cltdob_fix","f_purchase_lh"]

df = df[columns_to_keep]
df = df.fillna(0)

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
### FILTERING AND UPSAMPLING


### 4. TRAINING THE MODEL
## We selected Random Forest since it is robust to overfitting and tends to perform well without much hyperparameter tuning.
X = df[["flg_gi_claim","flg_is_proposal","is_housewife_retiree", "is_sg_pr", "is_class_1_2","annual_income_est","n_months_last_bought_products"
                   ,"flg_latest_being_lapse","recency_lapse", "recency_cancel","tot_inforce_pols","f_mindef_mha","recency_clmcon","recency_giclaim","cltdob_fix"]]
y = df["f_purchase_lh"]

# Split the dataset into training and testing sets with 20% for test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print("F1 score:", f1)
### TRAINING THE MODEL


### TEST DATA

# Age handling
hidden_data['cltdob_fix'] = hidden_data['cltdob_fix'].replace(to_replace='None', value=np.nan).dropna()
hidden_data['cltdob_fix'] = pd.to_datetime(hidden_data['cltdob_fix'], format='%Y-%m-%d')
hidden_data['cltdob_fix'] = hidden_data['cltdob_fix'].apply(calculate_age)

# Label encoding of income
hidden_data['annual_income_est']= label_encoder.fit_transform(hidden_data['annual_income_est'])

hidden_data['clttype'] = label_encoder.fit_transform(hidden_data['clttype']) 
columns_to_keep = ["flg_gi_claim","flg_is_proposal","is_housewife_retiree", "is_sg_pr", "is_class_1_2","annual_income_est","n_months_last_bought_products"
                   ,"flg_latest_being_lapse","recency_lapse", "recency_cancel","tot_inforce_pols","f_mindef_mha","recency_clmcon","recency_giclaim","cltdob_fix"]
hidden_data = hidden_data[columns_to_keep]
hidden_data = hidden_data.fillna(0)
result = rf_classifier.predict(hidden_data)


