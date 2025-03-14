import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from data_preprocessing import (
    load_data, preprocess_data, prepare_sequences, scale_data, split_data,
    visualize_hsi_data, perform_pca
)
from models import (
    create_lstm_model, create_cnn_residual_model, create_tcn_model,
    create_transformer_model, compile_model
)
import optuna
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping

def train_random_forest(X_train, y_train):
    """Train a Random Forest regression model."""
    print("\n=== Training Random Forest Model ===")
    
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create cross-validation object
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create Random Forest model
    rf = RandomForestRegressor(random_state=42)
    
    # Use GridSearchCV to find optimal hyperparameters
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, 
                             scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    rf_model = grid_search.best_estimator_
    
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best CV score (negative MSE): {grid_search.best_score_:.4f}")
    
    return rf_model

def train_xgboost(X_train, y_train):
    """Train an XGBoost regression model."""
    print("\n=== Training XGBoost Model ===")
    
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Create cross-validation object
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create XGBoost model
    xgb = XGBRegressor(random_state=42)
    
    # Use GridSearchCV to find optimal hyperparameters
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=cv, 
                             scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    xgb_model = grid_search.best_estimator_
    
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best CV score (negative MSE): {grid_search.best_score_:.4f}")
    
    return xgb_model

def evaluate_model(model, X_test, y_test, model_name, is_neural_network=False):
    """Evaluate model performance."""
    print(f"\n=== Evaluating {model_name} Model ===")
    
    if is_neural_network:
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_pred = model.predict(X_test_reshaped)
    else:
        y_pred = model.predict(X_test)
    
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual DON Concentration')
    plt.ylabel('Predicted DON Concentration')
    plt.title(f'{model_name}: Actual vs Predicted DON Concentration')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.annotate(f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}', xy=(0.05, 0.95),
                 xycoords='axes fraction', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                 verticalalignment='top')
    plt.savefig(f'figures/{model_name.lower().replace(" ", "_")}_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'model_name': model_name, 'mae': mae, 'rmse': rmse, 'r2': r2}

def objective(trial, model_type, X_train, y_train, X_val, y_val):
    """Optuna objective function for hyperparameter optimization."""
    if model_type == 'Random Forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
        }
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return mean_squared_error(y_val, y_pred)
    
    elif model_type == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        model = XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return mean_squared_error(y_val, y_pred)
    
    elif model_type in ['LSTM', 'CNN Residual', 'TCN', 'Transformer']:
        # Neural network hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'units': trial.suggest_int('units', 32, 256),
            'mlp_dim': trial.suggest_int('mlp_dim', 32, 256),
            'num_heads': trial.suggest_int('num_heads', 2, 8),
            'num_blocks': trial.suggest_int('num_blocks', 1, 4),  # For CNN and TCN
            'num_filters': trial.suggest_int('num_filters', 32, 128),  # For CNN and TCN
            'num_transformer_blocks': trial.suggest_int('num_transformer_blocks', 1, 4)  # For Transformer
        }
        
        # Reshape input data for neural networks
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        # Create and compile model
        if model_type == 'LSTM':
            model = create_lstm_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
        elif model_type == 'CNN Residual':
            model = create_cnn_residual_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
        elif model_type == 'TCN':
            model = create_tcn_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
        else:  # Transformer
            # Create a copy of params and rename num_blocks to num_transformer_blocks
            transformer_params = params.copy()
            transformer_params['num_transformer_blocks'] = transformer_params.pop('num_blocks')
            model = create_transformer_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=transformer_params)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        
        # Train model
        history = model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=50,
            batch_size=params['batch_size'],
            callbacks=[
                TFKerasPruningCallback(trial, 'val_loss'),
                tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
            ],
            verbose=0
        )
        
        return history.history['val_loss'][-1]

def optimize_hyperparameters(model_type, X_train, y_train, X_val, y_val, n_trials=50):
    """Run Optuna optimization for a given model type."""
    # Create study with early stopping
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Define callback for early stopping
    def callback(study, trial):
        if len(study.trials) > 6:  # Minimum number of trials before checking for early stopping
            last_6_trials = study.trials[-6:]
            best_value = min(t.value for t in last_6_trials if t.value is not None)
            current_value = trial.value
            
            if current_value is not None and current_value > best_value * 1.001:  # 0.1% improvement threshold
                study.stop()
    
    # Run optimization with early stopping
    study.optimize(
        lambda trial: objective(trial, model_type, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        callbacks=[callback],
        gc_after_trial=True
    )
    
    return study.best_params, study.best_value

def train_neural_network(model_type, X_train, y_train, X_val, y_val, params):
    """Train a neural network model with the given parameters."""
    print(f"\n=== Training {model_type} Model ===")
    
    # Reshape input data for neural networks
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    # Create and compile model
    if model_type == 'LSTM':
        model = create_lstm_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
    elif model_type == 'CNN Residual':
        model = create_cnn_residual_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
    elif model_type == 'TCN':
        model = create_tcn_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
    else:  # Transformer
        model = create_transformer_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    # Train model
    history = model.fit(
        X_train_reshaped, y_train,
        validation_data=(X_val_reshaped, y_val),
        epochs=300,  # Increased to 300 epochs
        batch_size=params['batch_size'],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
        ],
        verbose=1
    )
    
    return model, history

def train_model(model_type, X_train, y_train, params=None):
    """Train a model with given parameters."""
    if model_type == 'Random Forest':
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        return model
    
    elif model_type == 'XGBoost':
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        return model
    
    elif model_type == 'LSTM':
        # Reshape input data
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        model = create_lstm_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = model.fit(
            X_train_reshaped,
            y_train,
            epochs=300,  # Increased to 300 epochs
            batch_size=32,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
            verbose=0
        )
        return model, history.history
    
    elif model_type == 'CNN Residual':
        # Reshape input data
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        model = create_cnn_residual_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = model.fit(
            X_train_reshaped,
            y_train,
            epochs=300,  # Increased to 300 epochs
            batch_size=32,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
            verbose=0
        )
        return model, history.history
    
    elif model_type == 'TCN':
        # Reshape input data
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        model = create_tcn_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = model.fit(
            X_train_reshaped,
            y_train,
            epochs=300,  # Increased to 300 epochs
            batch_size=32,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
            verbose=0
        )
        return model, history.history
    
    elif model_type == 'Transformer':
        # Reshape input data
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        model = create_transformer_model(input_shape=(X_train.shape[1], 1), output_shape=1, params=params)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = model.fit(
            X_train_reshaped,
            y_train,
            epochs=300,  # Increased to 300 epochs
            batch_size=32,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
            verbose=0
        )
        return model, history.history
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_and_save_models(data_path, sequence_length=10):
    """Train and save all models with different PCA components."""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Load and preprocess data
    df = load_data(data_path)
    df = preprocess_data(df)
    
    # Visualize HSI data
    visualize_hsi_data(data_path)
    
    # Prepare features and target
    spectral_features = [str(i) for i in range(448)]
    
    # Verify all spectral features exist
    missing_features = [f for f in spectral_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing spectral features: {missing_features}")
    
    X = df[spectral_features].values
    y = df['vomitoxin_ppb'].values
    
    # Scale data
    X_scaled, X_scaler = scale_data(X, scaler_type='standard')
    
    # Perform PCA with different components
    pca_results = perform_pca(X_scaled, y, n_components_list=[3, 5, 10, 20])
    
    # Store all results
    all_results = {}
    
    # Train models for each PCA configuration
    for n_components, pca_data in pca_results.items():
        print(f"\n=== Training models with {n_components} PCA components ===")
        
        # Create directory for this PCA configuration
        pca_dir = f'models/pca_{n_components}'
        os.makedirs(pca_dir, exist_ok=True)
        
        X_pca = pca_data['X_pca']
        pca = pca_data['pca']
        
        # Save scaler and PCA for this configuration
        joblib.dump(X_scaler, f'{pca_dir}/scaler.pkl')
        joblib.dump(pca, f'{pca_dir}/pca.pkl')
        
        # Save PCA results
        pca_results_dict = {
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'mean': pca.mean_,
            'n_components': n_components
        }
        joblib.dump(pca_results_dict, f'{pca_dir}/pca_results.pkl')
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_pca, y)
        
        # Train and optimize each model
        model_types = ['Random Forest', 'XGBoost', 'LSTM', 'CNN Residual', 'TCN', 'Transformer']
        for model_type in model_types:
            print(f"\nOptimizing {model_type} model...")
            best_params, best_score = optimize_hyperparameters(model_type, X_train, y_train, X_val, y_val)
            print(f"Best parameters: {best_params}")
            print(f"Best validation score: {best_score:.4f}")
            
            # Train final model with best parameters
            if model_type in ['Random Forest', 'XGBoost']:
                model = train_model(model_type, X_train, y_train, best_params)
            else:
                model, history = train_neural_network(model_type, X_train, y_train, X_val, y_val, best_params)
                # Save training history
                joblib.dump(history.history, f'{pca_dir}/{model_type.lower().replace(" ", "_")}_history.pkl')
            
            # Evaluate model
            results = evaluate_model(model, X_test, y_test, f"{model_type} (PCA={n_components})", 
                                  is_neural_network=(model_type in ['LSTM', 'CNN Residual', 'TCN', 'Transformer']))
            
            # Save model
            if model_type in ['Random Forest', 'XGBoost']:
                joblib.dump(model, f'{pca_dir}/{model_type.lower().replace(" ", "_")}_model.pkl')
            else:
                model.save(f'{pca_dir}/{model_type.lower().replace(" ", "_")}_model.keras')
            
            # Store results
            all_results[n_components] = all_results.get(n_components, {})
            all_results[n_components][model_type] = results
    
    # Save all results
    joblib.dump(all_results, 'models/all_results.pkl')
    
    # Plot comparison of model performances
    plt.figure(figsize=(15, 8))
    metrics = ['mae', 'rmse', 'r2']
    model_names = ['Random Forest', 'XGBoost', 'LSTM', 'CNN Residual', 'TCN', 'Transformer']
    
    for metric in metrics:
        plt.subplot(1, 3, metrics.index(metric) + 1)
        for model_name in model_names:
            values = [all_results[n][model_name][metric] for n in [3, 5, 10, 20]]
            plt.plot([3, 5, 10, 20], values, marker='o', label=model_name)
        
        plt.xlabel('Number of PCA Components')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} vs PCA Components')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return all_results

if __name__ == "__main__":
    # Train models
    results = train_and_save_models('TASK-ML-INTERN.csv')
    print("\nAll models trained and saved successfully!") 