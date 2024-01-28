import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read
df = pd.read_parquet('catB_train.parquet')
# Step 1: Clean f_purchase_lh
df['f_purchase_lh'] = df['f_purchase_lh'].fillna(0)

### CORRELATION MATRIX
columns_to_test = ["clttype", "stat_flag", "flg_substandard", "flg_has_health_claim", "flg_is_borderline_standard"
                   , "flg_is_revised_term", "flg_is_rental_flat", "flg_has_life_claim", "flg_gi_claim", "flg_is_proposal"
                   , "flg_with_preauthorisation", "is_housewife_retiree", "is_sg_pr", "is_class_1_2"
                   , "f_purchase_lh", "hh_20", "pop_20", "hh_size", "hh_size_est"
                   , "annual_income_est"
                   , "n_months_last_bought_products", "tot_inforce_pols"]
# Test Correlation matrix
df = df[columns_to_test]
corr_matrix = df.corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
### CORRELATION MATRIX