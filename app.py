import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
from data_preprocessing import load_data, preprocess_data, visualize_hsi_data, perform_pca

# Set page config
st.set_page_config(
    page_title="DON Concentration Predictor",
    page_icon="🌽",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def load_models():
    models = {}
    scalers = {}
    pcas = {}
    histories = {}
    
    # Define custom objects for model loading
    custom_objects = {
        'MeanSquaredError': tf.keras.losses.MeanSquaredError,
        'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError
    }
    
    # Load models for each PCA configuration
    for n_components in [3, 5, 10, 20]:
        try:
            pca_dir = f'models/pca_{n_components}'
            
            # Load preprocessing objects
            scalers[n_components] = joblib.load(f'{pca_dir}/scaler.pkl')
            pcas[n_components] = joblib.load(f'{pca_dir}/pca.pkl')
            
            # Load machine learning models
            models[f'Random Forest (PCA={n_components})'] = joblib.load(f'{pca_dir}/random_forest_model.pkl')
            models[f'XGBoost (PCA={n_components})'] = joblib.load(f'{pca_dir}/xgboost_model.pkl')
            
            # Load neural network models with custom objects
            for model_type in ['LSTM', 'CNN Residual', 'TCN', 'Transformer']:
                model_name = f'{model_type} (PCA={n_components})'
                models[model_name] = tf.keras.models.load_model(
                    f'{pca_dir}/{model_type.lower().replace(" ", "_")}_model.keras',
                    custom_objects=custom_objects
                )
                # Load training history
                histories[model_name] = joblib.load(f'{pca_dir}/{model_type.lower().replace(" ", "_")}_history.pkl')
            
        except Exception as e:
            st.warning(f"Could not load models for PCA={n_components}: {str(e)}")
    
    return models, scalers, pcas, histories

# Load data
@st.cache_data
def load_and_preprocess_data():
    df = load_data('TASK-ML-INTERN.csv')
    df = preprocess_data(df)
    return df

def plot_predictions(actual, predictions, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=predictions, name='Predicted', line=dict(color='red')))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Value')
    return fig

def plot_model_comparison(results_df, metric):
    """Plot comparison of different models and PCA components."""
    fig = go.Figure()
    
    # Group by model type and PCA components
    for model_type in ['Random Forest', 'XGBoost', 'LSTM', 'CNN Residual', 'TCN', 'Transformer']:
        model_data = results_df[results_df['Model'].str.contains(model_type)]
        fig.add_trace(go.Scatter(
            x=model_data['PCA Components'],
            y=model_data[metric],
            name=model_type,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title=f'{metric.upper()} Comparison Across Models and PCA Components',
        xaxis_title='Number of PCA Components',
        yaxis_title=metric.upper(),
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def plot_feature_importance(model, feature_names, title):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=feature_names,
            y=importance,
            marker_color='teal'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Features',
        yaxis_title='Importance',
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def plot_learning_curves(history, title):
    """Plot learning curves for neural network models."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=history['loss'],
        name='Training Loss',
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        y=history['val_loss'],
        name='Validation Loss',
        mode='lines'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Epoch',
        yaxis_title='Loss',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def plot_training_history(history, model_name):
    """Plot training history for neural network models."""
    fig = go.Figure()
    
    # Plot training loss
    fig.add_trace(go.Scatter(
        y=history['loss'],
        name='Training Loss',
        mode='lines',
        line=dict(color='blue')
    ))
    
    # Plot validation loss
    fig.add_trace(go.Scatter(
        y=history['val_loss'],
        name='Validation Loss',
        mode='lines',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=f'{model_name} Training History',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def plot_pca_analysis(n_components):
    """Plot comprehensive PCA analysis."""
    try:
        # Load PCA analysis results
        pca_results = joblib.load(f'models/pca_{n_components}/pca_results.pkl')
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["2D Visualization", "3D Visualization", "Component Analysis"])
        
        with tab1:
            st.image(f"figures/pca_2d_{n_components}.png", 
                    caption=f"2D PCA Visualization with {n_components} Components")
            st.markdown("""
            This 2D visualization shows the first two principal components, 
            with colors representing DON concentration. This helps identify 
            any clustering patterns in the data.
            """)
        
        with tab2:
            st.image(f"figures/pca_3d_{n_components}.png", 
                    caption=f"3D PCA Visualization with {n_components} Components")
            st.markdown("""
            The 3D visualization provides a more detailed view of the data 
            distribution across the first three principal components.
            """)
        
        with tab3:
            st.image(f"figures/pca_component_analysis_{n_components}.png", 
                    caption=f"Component Analysis for {n_components} Components")
            st.markdown("""
            This analysis shows:
            - Left: Correlation between each component and DON concentration
            - Right: Explained variance ratio for each component
            """)
            
            # Display numerical results
            st.subheader("Component Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Explained Variance Ratios:")
                for i, var in enumerate(pca_results['explained_variance']):
                    st.write(f"PC{i+1}: {var:.4f}")
            with col2:
                st.write("Cumulative Variance:")
                for i, var in enumerate(pca_results['cumulative_variance']):
                    st.write(f"Up to PC{i+1}: {var:.4f}")
    
    except Exception as e:
        st.warning(f"PCA analysis plots not available: {str(e)}")

def plot_model_comparison_analysis():
    """Plot comprehensive model comparison analysis."""
    try:
        # Load all results
        all_results = joblib.load('models/all_results.pkl')
        
        # Create tabs for different metrics
        tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Learning Curves", "Error Analysis"])
        
        with tab1:
            st.image("figures/model_comparison.png", 
                    caption="Model Performance Comparison Across PCA Components")
            st.markdown("""
            This plot shows how different models perform across various metrics 
            (MAE, RMSE, R²) for different numbers of PCA components.
            """)
            
            # Display numerical results
            st.subheader("Best Model Performance")
            best_models = {}
            for n_components in [3, 5, 10, 20]:
                best_model = min(all_results[n_components].items(), 
                               key=lambda x: x[1]['rmse'])
                best_models[n_components] = best_model
            
            for n_components, (model_name, metrics) in best_models.items():
                st.write(f"\nPCA Components: {n_components}")
                st.write(f"Best Model: {model_name}")
                st.write(f"RMSE: {metrics['rmse']:.4f}")
                st.write(f"MAE: {metrics['mae']:.4f}")
                st.write(f"R²: {metrics['r2']:.4f}")
        
        with tab2:
            st.subheader("Learning Curves for Neural Network Models")
            model_type = st.selectbox(
                "Select Model Type",
                ['LSTM', 'CNN Residual', 'TCN', 'Transformer']
            )
            
            # Load training histories
            histories = {}
            for n_components in [3, 5, 10, 20]:
                model_name = f"{model_type} (PCA={n_components})"
                try:
                    histories[model_name] = joblib.load(f'models/pca_{n_components}/{model_type.lower().replace(" ", "_")}_history.pkl')
                except Exception as e:
                    st.warning(f"Could not load history for {model_name}: {str(e)}")
                    continue
            
            # Plot learning curves for different PCA components
            fig = go.Figure()
            for n_components in [3, 5, 10, 20]:
                model_name = f"{model_type} (PCA={n_components})"
                if model_name in histories:
                    history = histories[model_name]
                    fig.add_trace(go.Scatter(
                        y=history['val_loss'],
                        name=f'PCA={n_components}',
                        mode='lines'
                    ))
            
            fig.update_layout(
                title=f"{model_type} Learning Curves",
                xaxis_title='Epoch',
                yaxis_title='Validation Loss',
                showlegend=True,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Error Analysis")
            # Plot error distribution
            fig = go.Figure()
            for n_components in [3, 5, 10, 20]:
                for model_name, metrics in all_results[n_components].items():
                    fig.add_trace(go.Box(
                        y=[metrics['rmse']],
                        name=f"{model_name} (PCA={n_components})",
                        boxpoints='all'
                    ))
            
            fig.update_layout(
                title="Error Distribution Across Models",
                yaxis_title='RMSE',
                showlegend=False,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Model comparison analysis not available: {str(e)}")

def main():
    st.title("Hyperspectral Imaging DON Concentration Predictor")
    
    # Display email in big words
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;'>
        <h2 style='color: #1f77b4;'>Contact: devanshgupta049@gmail.com</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts Deoxynivalenol (DON) concentration in corn samples using hyperspectral images.
    
    Features:
    - Multiple model comparison
    - Interactive visualizations
    - Model performance metrics
    - Comprehensive PCA analysis
    - Training history visualization
    - Optuna hyperparameter optimization
    """)
    
    # Sidebar
    st.sidebar.header("Model Configuration")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Choose a model",
        ["Random Forest", "XGBoost", "LSTM", "CNN Residual", "TCN", "Transformer"]
    )
    
    # PCA components selection
    n_components = st.sidebar.selectbox(
        "Number of PCA Components",
        [3, 5, 10, 20]
    )
    
    # Load data and models
    try:
        df = load_and_preprocess_data()
        models, scalers, pcas, histories = load_models()
        models_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        models_loaded = False
    
    # Data Overview
    st.header("Data Overview")
    
    # Display data statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Interactive Data Visualization
    st.subheader("Interactive Data Analysis")
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["DON Distribution", "Feature Correlation", "Sample Distribution"]
    )
    
    if viz_type == "DON Distribution":
        fig = px.histogram(df, x='vomitoxin_ppb', 
                          title='Distribution of DON Concentration',
                          labels={'vomitoxin_ppb': 'DON Concentration (ppb)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics
        st.write("Summary Statistics for DON Concentration:")
        st.write(df['vomitoxin_ppb'].describe())
    
    elif viz_type == "Feature Correlation":
        # Create correlation matrix for first 20 features
        corr_matrix = df[[str(i) for i in range(20)] + ['vomitoxin_ppb']].corr()
        fig = px.imshow(corr_matrix, 
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Sample Distribution":
        fig = px.scatter(df, x='vomitoxin_ppb', y='0',
                        title='Sample Distribution',
                        labels={'vomitoxin_ppb': 'DON Concentration (ppb)',
                               '0': 'First Spectral Feature'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Best Models Section
    st.header("Best Performing Models")
    try:
        # Load all results
        all_results = joblib.load('models/all_results.pkl')
        
        # Create a DataFrame for best models
        best_models_data = []
        for n_components in [3, 5, 10, 20]:
            best_model = min(all_results[n_components].items(), 
                           key=lambda x: x[1]['rmse'])
            best_models_data.append({
                'PCA Components': n_components,
                'Model': best_model[0],
                'RMSE': best_model[1]['rmse'],
                'MAE': best_model[1]['mae'],
                'R²': best_model[1]['r2']
            })
        
        best_models_df = pd.DataFrame(best_models_data)
        
        # Display best models in a table
        st.dataframe(best_models_df.style.format({
            'RMSE': '{:.4f}',
            'MAE': '{:.4f}',
            'R²': '{:.4f}'
        }))
        
        # Create interactive plot for best models
        fig = go.Figure()
        for metric in ['RMSE', 'MAE', 'R²']:
            fig.add_trace(go.Scatter(
                x=best_models_df['PCA Components'],
                y=best_models_df[metric],
                name=metric,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title='Best Model Performance Across PCA Components',
            xaxis_title='Number of PCA Components',
            yaxis_title='Metric Value',
            showlegend=True,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not load best models data: {str(e)}")
    
    # PCA Analysis Section
    st.header("PCA Analysis")
    plot_pca_analysis(n_components)
    
    # Model Comparison Section
    st.header("Model Comparison")
    plot_model_comparison_analysis()
    
    # Model Selection and Prediction
    st.header("Model Selection and Prediction")
    
    if models_loaded:
        # Sample selection by hsi_id
        st.subheader("Sample Selection")
        selected_hsi_id = st.selectbox(
            "Select HSI ID",
            df['hsi_id'].unique()
        )
        
        # Make predictions
        if st.button("Generate Predictions"):
            try:
                # Get data for selected hsi_id
                sample_data = df[df['hsi_id'] == selected_hsi_id]
                if len(sample_data) == 0:
                    st.error("Selected HSI ID not found in the dataset")
                    return
                
                # Get spectral features
                spectral_features = [str(i) for i in range(448)]
                X = sample_data[spectral_features].values
                
                # Scale the features
                X_scaled = scalers[n_components].transform(X)
                
                # Apply PCA transformation
                X_pca = pcas[n_components].transform(X_scaled)
                
                # Select model
                model_name = f"{selected_model} (PCA={n_components})"
                model = models[model_name]
                
                # Generate predictions
                if selected_model in ['LSTM', 'CNN Residual', 'TCN', 'Transformer']:
                    X_reshaped = X_pca.reshape(X_pca.shape[0], X_pca.shape[1], 1)
                    predictions = model.predict(X_reshaped).ravel()
                else:
                    predictions = model.predict(X_pca)
                
                # Display results
                st.subheader("Prediction Results")
                actual_value = sample_data['vomitoxin_ppb'].values[0]
                predicted_value = predictions[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Actual DON Concentration", f"{actual_value:.2f} ppb")
                with col2:
                    st.metric("Predicted DON Concentration", f"{predicted_value:.2f} ppb")
                
                # Plot predictions vs actual
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=[actual_value],
                    name='Actual',
                    mode='markers',
                    marker=dict(size=15, color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=[predicted_value],
                    name='Predicted',
                    mode='markers',
                    marker=dict(size=15, color='red')
                ))
                fig.update_layout(
                    title=f"{model_name} Prediction",
                    yaxis_title="DON Concentration (ppb)",
                    showlegend=True,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Model performance metrics
                st.subheader("Model Performance")
                mse = (actual_value - predicted_value) ** 2
                mae = abs(actual_value - predicted_value)
                rmse = np.sqrt(mse)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MSE", f"{mse:.4f}")
                with col2:
                    st.metric("MAE", f"{mae:.4f}")
                with col3:
                    st.metric("RMSE", f"{rmse:.4f}")
                
                # Show training history for neural network models
                if selected_model in ['LSTM', 'CNN Residual', 'TCN', 'Transformer']:
                    st.subheader("Training History")
                    history = histories[model_name]
                    history_fig = plot_training_history(history, model_name)
                    st.plotly_chart(history_fig, use_container_width=True)
                
                # Download prediction results
                results_df = pd.DataFrame({
                    'hsi_id': [selected_hsi_id],
                    'Actual_DON': [actual_value],
                    'Predicted_DON': [predicted_value],
                    'Error': [mae]
                })
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="don_prediction.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")

if __name__ == "__main__":
    main() 