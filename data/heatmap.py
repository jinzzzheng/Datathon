import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from datetime import date

# read
df = pd.read_parquet('/Users/yujinzheng/Downloads/cs2040/Datathon/data/catB_train.parquet')
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
                   ,"flg_latest_being_lapse","recency_lapse", "recency_cancel","tot_inforce_pols","f_mindef_mha","recency_clmcon","recency_giclaim","f_purchase_lh"]
# Test Correlation matrix
df = df[columns_to_test]
corr_matrix = df.corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
### CORRELATION MATRIX
