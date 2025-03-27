import pandas as pd
import numpy as np

# Parameters
num_samples = 500  # Number of individuals
num_snps = 100  # Number of SNPs (reduced for better correlation)

# Set random seed for reproducibility
np.random.seed(42)

# Generate phenotype data with stronger correlation to certain SNPs
phenotype_data = pd.DataFrame({
    'sample_id': [f'sample_{i}' for i in range(1, num_samples + 1)],
    'phenotype': np.random.normal(100, 15, num_samples)  # Mean = 100, SD = 15
})

# Introduce genetic influence (SNP1, SNP5, SNP8 strongly affect phenotype)
for snp in [1, 5, 8]:
    influence = np.random.normal(5, 2, num_samples) * np.random.choice([0, 1, 2], num_samples)
    phenotype_data['phenotype'] += influence

# Generate genotype data efficiently using NumPy
snp_ids = [f'SNP{snp}' for snp in range(1, num_snps + 1)]
genotype_matrix = np.random.choice([0, 1, 2], size=(num_samples, num_snps))

# Convert to DataFrame efficiently
genotype_data = pd.DataFrame(genotype_matrix, columns=snp_ids)
genotype_data.insert(0, 'sample_id', [f'sample_{i}' for i in range(1, num_samples + 1)])  # Add sample_id

# Save data to CSV
genotype_path = "genotype_data.csv"
phenotype_path = "phenotype_data.csv"

genotype_data.to_csv(genotype_path, index=False)
phenotype_data.to_csv(phenotype_path, index=False)

print(f"Genotype data saved to: {genotype_path}")
print(f"Phenotype data saved to: {phenotype_path}")

