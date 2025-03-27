import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from scipy.stats import probplot

# File paths
genotype_file_path = "genotype_data.csv"
phenotype_file_path = "phenotype_data.csv"

# Load genotype and phenotype data
genotype_data = pd.read_csv(genotype_file_path)
phenotype_data = pd.read_csv(phenotype_file_path)

# Merge datasets on 'sample_id'
combined_data = pd.merge(genotype_data, phenotype_data, on='sample_id')

# Preprocessing: Log-transform phenotype if skewed
if combined_data['phenotype'].skew() > 1:
    combined_data['phenotype'] = np.log(combined_data['phenotype'])

# Prepare data for PCA
X = combined_data.drop(columns=['sample_id', 'phenotype'])
y = combined_data['phenotype']

# Perform PCA
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X)

# Save PCA-transformed data
np.save("X_pca.npy", X_pca)

# Add constant to PCA components for regression
X_pca = sm.add_constant(X_pca)

# Perform OLS regression
model = sm.OLS(y, X_pca).fit()

# Display regression summary
print(model.summary())

# Extract p-values and correct for multiple testing
p_values = model.pvalues[1:]  # Exclude constant term
_, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

# Create Manhattan plot
plt.figure(figsize=(12, 6))
plt.scatter(range(len(corrected_p_values)), -np.log10(corrected_p_values), color='blue')
plt.xlabel('PCA Component Index')
plt.ylabel('-log10(corrected p-value)')
plt.title('Manhattan Plot')
plt.tight_layout()
plt.show()

# Create QQ plot
plt.figure(figsize=(12, 6))
probplot(-np.log10(corrected_p_values), dist="norm", plot=plt)
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Observed Quantiles')
plt.title('QQ Plot')
plt.tight_layout()
plt.show()

# Correlation Matrix of PCA components
corr_matrix = pd.DataFrame(X_pca).corr()

# Plot correlation matrix as heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix (PCA Components vs Phenotype)')
plt.tight_layout()
plt.show()

