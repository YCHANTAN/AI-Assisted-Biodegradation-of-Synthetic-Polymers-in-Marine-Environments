AI-Assisted Biodegradation of Synthetic Polymers in Marine Environments

This project utilizes Machine Learning (K-Means Clustering) and Chemical Informatics (RDKit) to analyze the biodegradation potential of synthetic polymers like PVA, PEG, and PAA in marine ecosystems. By transforming SMILES strings into numerical descriptors, the system identifies structural "hotspots" that are most susceptible to microbial enzymatic attack.

🛠️ PrerequisitesBefore running the script, ensure you have Python 3.12+ installed and the following libraries:Bashpip install pandas matplotlib seaborn rdkit scikit-learn openpyxl

📂 Project StructureDatasets-3.xlsx: The primary dataset containing polymer names, SMILES strings, and Molecular Weights polymer_analysis.py: The main AI pipeline for feature extraction, scaling, and clustering.Outputs:polymer_mw_distribution.png: Histogram of the dataset's molecular weight range.polymer_correlation_heatmap.png: Statistical relationship between chemical features.polymer_ai_clusters.png: Visual representation of AI-grouped polymer classes.polymer_biodegradation_indicators.png: Scatter plot of $M_w$ vs. $LogP$ (Hydrophobicity).

🚀 How to RunPlace Datasets-3.xlsx and polymer_analysis.py in the same folder.Open your terminal or VS Code.Run the script:Bashpython polymer_analysis.py

📊 Key Research Findings
1. The PVA PotentialThe analysis identified Poly(vinyl alcohol) (PVA) as the molecule with the highest potential for marine bioremediation due to its high water solubility and accessible hydroxyl (-OH) functional groups.
2. The Chain-Length ParadoxThe AI models confirmed that as Molecular Weight increases, the predicted biodegradability decreases exponentially, creating a "refractory" cluster of polymers that persist longer in oceanic environments.
3. AI ClusteringThe unsupervised K-Means model successfully categorized polymers into 4 risk tiers:Green (Cluster 1): Highly degradable, low. Yellow: Intermediate persistence.Red: High refractory materials (e.g., high-density PAA).

🧪 Methodology
1. Feature Engineering: SMILES strings are converted to 7 quantitative descriptors (LogP, TPSA, Rotatable Bonds, etc.).
2. Scaling: Data is normalized using StandardScaler to ensure the AI isn't biased by units.
3. Clustering: K-Means identifies hidden patterns in chemical structural data.
4. Visualization: Principal Component Analysis (PCA) is used to project high-dimensional data into a readable 2D format.
