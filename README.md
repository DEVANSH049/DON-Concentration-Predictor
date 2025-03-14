# DON Concentration Prediction in Corn Using Hyperspectral Imaging

## Overview
This project develops a machine learning-based system for predicting Deoxynivalenol (DON) concentration in corn samples using hyperspectral imaging data. The system employs both traditional machine learning and deep learning approaches to achieve accurate predictions, with the best model achieving an RMSE of 1423.12 ppb.

## Video Demonstration
Watch the interactive Streamlit app demonstration:CLICK IT
[![DON Prediction App Demo](https://img.youtube.com/vi/JKl-dyVAp0A/0.jpg)](https://www.youtube.com/watch?v=JKl-dyVAp0A)

## Features
- **Multiple Model Architectures**:
  - Traditional ML: Random Forest, XGBoost
  - Deep Learning: LSTM, CNN Residual, TCN, Transformer
- **Dimensionality Reduction**: PCA with 3, 5, 10, and 20 components
- **Interactive Web Interface**: Streamlit-based application
- **Comprehensive Analysis**: Including feature importance, error analysis, and model comparison
- **Real-time Prediction**: Instant DON concentration estimation

## Project Structure
```
├── app.py                 # Streamlit web application
├── models/               # Trained model files
│   ├── all_results.pkl   # Combined model results
│   └── pca_*/           # PCA-specific model results
├── figures/             # Generated visualization plots
├── data/               # Dataset and preprocessing scripts
├── train_models.py     # Model training script
└── analyze_results.py  # Results analysis script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/don-prediction.git
cd don-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload hyperspectral data or use the sample data provided

4. Select model and PCA configuration

5. View predictions and analysis results

## Model Performance

### Best Models by PCA Configuration

| PCA Components | Model | RMSE (ppb) | MAE (ppb) | R² |
|---------------|-------|------------|-----------|-----|
| 3 | XGBoost | 1588.34 | 835.09 | 0.9821 |
| 5 | CNN Residual | 1487.65 | 792.43 | 0.9883 |
| 10 | XGBoost | 1423.12 | 781.87 | 0.9901 |
| 20 | Transformer | 1502.31 | 803.56 | 0.9876 |

### Training Characteristics

- **LSTM**: 80-120 epochs, moderate convergence
- **CNN Residual**: 50-70 epochs, fast convergence
- **TCN**: Slower initial convergence, stable patterns
- **Transformer**: Variable convergence, 40-100 epochs

## Data Preprocessing

1. **Standardization**:
   - Applied StandardScaler to normalize spectral bands
   - Zero mean and unit variance transformation

2. **Dimensionality Reduction**:
   - PCA with 3-20 components
   - First component captures 87.08% of variance
   - First 3 components capture 95.04% of variance

3. **Target Transformation**:
   - Log transformation of DON concentration
   - Addresses right-skewed distribution

## Model Architecture Details

### Traditional ML Models

#### Random Forest
- Base estimators: Decision trees
- Ensemble method: Bagging
- Hyperparameters:
  - n_estimators: 50-300
  - max_depth: 5-30
  - min_samples_split: 2-20

#### XGBoost
- Base estimators: Decision trees
- Ensemble method: Gradient boosting
- Hyperparameters:
  - learning_rate: 0.01-0.3
  - max_depth: 3-12
  - n_estimators: 50-500
  - subsample: 0.5-1.0

### Deep Learning Models

#### LSTM
- Architecture:
  - 2-3 LSTM layers (64-256 units)
  - Dropout rate: 0.2-0.5
  - Dense output layer

#### CNN Residual
- Architecture:
  - 1-4 residual blocks
  - 32-128 filters per block
  - Skip connections

#### TCN
- Architecture:
  - 1-4 TCN blocks
  - Dilated causal convolutions
  - Exponential dilation rates

#### Transformer
- Architecture:
  - 1-4 transformer blocks
  - 2-8 attention heads
  - Feed-forward dimension: 32-256

## Error Analysis

### Model-Specific Error Patterns

| Model | Mean Error (ppb) | Std Dev (ppb) | Range (ppb) |
|-------|-----------------|---------------|-------------|
| Random Forest | 42.81 | 1721.56 | [-5823.94, 7125.35] |
| XGBoost | 28.35 | 1402.78 | [-4562.33, 5934.12] |
| LSTM | 35.27 | 1594.63 | [-5102.45, 6328.73] |
| CNN Residual | 31.82 | 1478.92 | [-4823.56, 6045.18] |
| TCN | 39.45 | 1538.27 | [-4912.38, 6187.24] |
| Transformer | 30.18 | 1485.64 | [-4753.29, 6112.47] |

## Future Improvements

1. **Model Enhancement**:
   - Explore ensemble methods
   - Investigate attention mechanisms
   - Develop specialized architectures

2. **Data Collection**:
   - Expand dataset size
   - Include diverse corn varieties
   - Collect temporal data

3. **Deployment Optimization**:
   - Model compression
   - Mobile-friendly deployment
   - Real-time monitoring

4. **Validation and Testing**:
   - Field trials
   - Traditional method comparison
   - Standardized protocols

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or feedback, please contact:
- Email: [devanshgupta049@gmail.com]
- GitHub: [[your-github-profile](https://github.com/DEVANSH049)] 
