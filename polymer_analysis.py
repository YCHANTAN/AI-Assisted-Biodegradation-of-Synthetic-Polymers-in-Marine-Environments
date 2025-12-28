import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# 1. LOAD DATA (Updated for Excel)
filename = 'Datasets-3.xlsx'

if not os.path.exists(filename):
    print(f"Error: The file '{filename}' was not found in this folder.")
else:
    print(f"Loading {filename}...")
    # read_excel handles symbols and encodings automatically
    df = pd.read_excel(filename)

    # 2. FEATURE EXTRACTION (AI PREPROCESSING)
    def get_features(smiles):
        s = str(smiles).replace('*', '')
        mol = Chem.MolFromSmiles(s)
        if mol:
            return {
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'HDonors': Descriptors.NumHDonors(mol),
                'HAcceptors': Descriptors.NumHAcceptors(mol),
                'MonomerWt': Descriptors.ExactMolWt(mol)
            }
        return None

    print("Extracting chemical features using RDKit...")
    # Apply features calculation
    feat_list = [get_features(s) for s in df['Smiles']]
    desc_df = pd.DataFrame([f if f is not None else {k: np.nan for k in ['LogP', 'TPSA', 'RotBonds', 'HDonors', 'HAcceptors', 'MonomerWt']} for f in feat_list])
    
    # Merge and clean
    clean_df = pd.concat([df.reset_index(drop=True), desc_df], axis=1).dropna(subset=['LogP', 'Mw'])

    # 3. AI MODELING (CLUSTERING)
    print("Running AI Clustering...")
    analysis_cols = ['Mw', 'LogP', 'TPSA', 'RotBonds', 'HDonors', 'HAcceptors', 'MonomerWt']
    X_scaled = StandardScaler().fit_transform(clean_df[analysis_cols])

    # K-Means AI Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clean_df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for 2D Visualization
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    clean_df['PC1'], clean_df['PC2'] = pcs[:, 0], pcs[:, 1]

    # 4. GENERATING CHARTS
    print("Generating and saving charts...")

    # Chart 1: Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['Mw'].dropna(), bins=20, color='teal', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Polymer Molecular Weights')
    plt.xlabel('Molecular Weight (Da)')
    plt.xscale('log')
    plt.savefig('polymer_mw_distribution.png')
    plt.close()

    # Chart 2: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(clean_df[analysis_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Chemical Property Correlation')
    plt.savefig('polymer_correlation_heatmap.png')
    plt.close()

    # Chart 3: AI Clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=clean_df, x='PC1', y='PC2', hue='Cluster', style='Substance Name', palette='bright', s=100)
    plt.title('AI Clustering of Polymers (Based on Molecular Features)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small', ncol=1)
    plt.tight_layout()
    plt.savefig('polymer_ai_clusters.png')
    plt.close()

    # Chart 4: Biodegradation Indicators
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=clean_df, x='Mw', y='LogP', hue='Substance Name', s=80)
    plt.xscale('log')
    plt.title('Environmental Predictors: Molecular Weight vs. Hydrophobicity')
    plt.xlabel('Polymer Molecular Weight (Mw)')
    plt.ylabel('Calculated LogP')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    plt.tight_layout()
    plt.savefig('polymer_biodegradation_indicators.png')
    plt.close()

    print("Success! All charts have been saved in your project folder.")