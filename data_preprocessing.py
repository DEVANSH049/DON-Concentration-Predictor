import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
import os

def load_data(file_path):
    """Load and prepare the dataset."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data including handling missing values and feature engineering."""
    # Create a copy to avoid fragmentation warnings
    df = df.copy()
    
    # Handle missing values
    df = df.ffill().bfill()
    
    # Get spectral features (0-447)
    spectral_features = [str(i) for i in range(448)]
    
    # Ensure all spectral features exist
    missing_features = [f for f in spectral_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing spectral features: {missing_features}")
    
    # Create rolling statistics for spectral features only
    rolling_stats = {}
    for col in spectral_features:
        rolling_stats[f'{col}_rolling_mean'] = df[col].rolling(window=5).mean()
        rolling_stats[f'{col}_rolling_std'] = df[col].rolling(window=5).std()
    
    # Convert rolling statistics to DataFrame and combine with original data
    rolling_stats_df = pd.DataFrame(rolling_stats)
    df = pd.concat([df, rolling_stats_df], axis=1)
    
    # Fill any remaining NaN values in rolling statistics
    df = df.bfill().ffill()
    
    return df

def prepare_sequences(data, sequence_length):
    """Prepare sequences for time series prediction."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def scale_data(data, scaler_type='minmax'):
    """Scale the data using specified scaler."""
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def split_data(X, y, test_size=0.2, val_size=0.2):
    """Split data into train, validation, and test sets."""
    # First split: train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Second split: train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size/(1-test_size), random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def visualize_hsi_data(csv_path):
    """Visualize hyperspectral imaging data."""
    # Create directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # The spectral columns are labeled as strings '0' through '447'
    spectral_cols = [str(i) for i in range(448)]
    
    # Extract the spectral data (X) and the vomitoxin values (y)
    X = df[spectral_cols].values
    y = df['vomitoxin_ppb'].values
    
    # Treat each spectral column index as a "wavelength" index
    wavelengths = np.arange(X.shape[1])
    
    # 1. Average spectral signature
    plt.figure(figsize=(10, 5))
    mean_spectrum = X.mean(axis=0)
    std_spectrum = X.std(axis=0)
    
    plt.plot(wavelengths, mean_spectrum, label='Mean Reflectance', color='blue')
    plt.fill_between(
        wavelengths,
        mean_spectrum - std_spectrum,
        mean_spectrum + std_spectrum,
        alpha=0.2,
        color='blue',
        label='±1 Std Dev'
    )
    plt.title('Average Spectral Signature')
    plt.xlabel('Spectral Band Index')
    plt.ylabel('Reflectance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('figures/average_spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Random subset of spectral signatures
    plt.figure(figsize=(10, 5))
    subset_size = min(20, len(df))
    sample_indices = np.random.choice(len(df), size=subset_size, replace=False)
    
    norm = plt.Normalize(y.min(), y.max())
    cmap = plt.cm.viridis
    
    for i in sample_indices:
        plt.plot(wavelengths, X[i, :], color=cmap(norm(y[i])), alpha=0.7)
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('vomitoxin_ppb')
    
    plt.title('Random Subset of Spectral Signatures Colored by vomitoxin_ppb')
    plt.xlabel('Spectral Band Index')
    plt.ylabel('Reflectance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/spectral_signatures_by_vomitoxin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation plot
    plt.figure(figsize=(10, 5))
    correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
    plt.bar(wavelengths, correlations, alpha=0.7, color='teal')
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    plt.title('Correlation of Each Spectral Band with vomitoxin_ppb')
    plt.xlabel('Spectral Band Index')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/correlation_with_vomitoxin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap
    plt.figure(figsize=(10, 6))
    subset_size = min(30, len(df))
    subset_indices = np.random.choice(len(df), size=subset_size, replace=False)
    sorted_indices = subset_indices[np.argsort(y[subset_indices])]
    heatmap_data = X[sorted_indices, :]
    
    sns.heatmap(heatmap_data, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('Heatmap of Spectral Signatures (Sorted by vomitoxin_ppb)')
    plt.xlabel('Spectral Bands')
    plt.ylabel('Samples')
    plt.savefig('figures/spectral_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_pca(X, y, n_components_list=[3, 5, 10, 20]):
    """Perform PCA with different numbers of components and visualize results."""
    pca_results = {}
    
    # Create figures directory if it doesn't exist
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Fit PCA with maximum components to get explained variance
    pca_full = PCA()
    pca_full.fit(X)
    
    # Plot cumulative explained variance
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs Number of Components')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/pca_cumulative_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot individual component variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
            pca_full.explained_variance_ratio_)
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Component')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/pca_component_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Perform PCA for each number of components
    for n_components in n_components_list:
        print(f"\nPerforming PCA with {n_components} components...")
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"Explained variance ratio: {explained_variance}")
        print(f"Cumulative explained variance: {cumulative_variance}")
        
        # Visualize PCA results
        if n_components >= 2:
            # 2D scatter plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
            plt.colorbar(scatter, label='DON Concentration (ppb)')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title(f'PCA Visualization (2D) - {n_components} Components')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f'figures/pca_2d_{n_components}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        if n_components >= 3:
            # 3D scatter plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                               c=y, cmap='viridis')
            plt.colorbar(scatter, label='DON Concentration (ppb)')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            ax.set_zlabel('Third Principal Component')
            plt.title(f'PCA Visualization (3D) - {n_components} Components')
            plt.savefig(f'figures/pca_3d_{n_components}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Store results
        pca_results[n_components] = {
            'X_pca': X_pca,
            'pca': pca,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance
        }
    
    return pca_results

def analyze_pca_components(X, y, n_components_list=[3, 5, 10, 20]):
    """Analyze the importance of PCA components in predicting DON concentration."""
    analysis_results = {}
    
    for n_components in n_components_list:
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Calculate correlation between components and target
        correlations = []
        for i in range(n_components):
            correlation = np.corrcoef(X_pca[:, i], y)[0, 1]
            correlations.append(correlation)
        
        # Store results
        analysis_results[n_components] = {
            'correlations': correlations,
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
        
        # Plot component importance
        plt.figure(figsize=(12, 6))
        
        # Plot correlations
        plt.subplot(1, 2, 1)
        plt.bar(range(1, n_components + 1), correlations)
        plt.xlabel('Component')
        plt.ylabel('Correlation with DON')
        plt.title(f'Component Correlations (PCA={n_components})')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot explained variance
        plt.subplot(1, 2, 2)
        plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
        plt.xlabel('Component')
        plt.ylabel('Explained Variance')
        plt.title(f'Component Variance (PCA={n_components})')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'figures/pca_component_analysis_{n_components}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return analysis_results 