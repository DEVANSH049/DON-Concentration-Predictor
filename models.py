import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut connection
        self.shortcut = layers.Conv1D(filters, 1, strides=strides) if strides != 1 else layers.Identity()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        shortcut = self.shortcut(inputs)
        x = layers.Add()([x, shortcut])
        return tf.nn.relu(x)

class TCNBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1):
        super(TCNBlock, self).__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.bn2 = layers.BatchNormalization()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        return tf.nn.relu(x)

def create_transformer_model(input_shape, output_shape, params=None):
    """Create a Transformer model with configurable hyperparameters."""
    if params is None:
        params = {
            'num_heads': 4,
            'num_transformer_blocks': 2,
            'mlp_dim': 64,
            'dropout_rate': 0.2
        }
    
    def transformer_block(x, num_heads, mlp_dim, dropout_rate):
        # Multi-head attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=mlp_dim // num_heads
        )(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # MLP
        mlp_output = tf.keras.layers.Dense(mlp_dim, activation='relu')(x)
        mlp_output = tf.keras.layers.Dense(x.shape[-1])(mlp_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + mlp_output)
        
        return x
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Positional encoding
    x = tf.keras.layers.Dense(params['mlp_dim'])(inputs)
    
    # Transformer blocks
    for _ in range(params['num_transformer_blocks']):
        x = transformer_block(x, params['num_heads'], params['mlp_dim'], params['dropout_rate'])
        x = tf.keras.layers.Dropout(params['dropout_rate'])(x)
    
    # Global average pooling and output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(output_shape)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_lstm_model(input_shape, output_shape, params=None):
    """Create an LSTM model with configurable hyperparameters."""
    if params is None:
        params = {
            'num_layers': 2,
            'units': 64,
            'dropout_rate': 0.2
        }
    
    model = tf.keras.Sequential()
    
    # Add LSTM layers
    for i in range(params['num_layers']):
        if i == 0:
            model.add(tf.keras.layers.LSTM(params['units'], return_sequences=(i < params['num_layers']-1),
                                         input_shape=input_shape))
        else:
            model.add(tf.keras.layers.LSTM(params['units'], return_sequences=(i < params['num_layers']-1)))
        model.add(tf.keras.layers.Dropout(params['dropout_rate']))
    
    # Output layer
    model.add(tf.keras.layers.Dense(output_shape))
    
    return model

def create_cnn_residual_model(input_shape, output_shape, params=None):
    """Create a CNN Residual model with configurable hyperparameters."""
    if params is None:
        params = {
            'num_filters': 64,
            'num_blocks': 3,
            'dropout_rate': 0.2
        }
    
    def residual_block(x, filters):
        shortcut = x
        x = tf.keras.layers.Conv1D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(params['num_filters'], 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    for _ in range(params['num_blocks']):
        x = residual_block(x, params['num_filters'])
        x = tf.keras.layers.Dropout(params['dropout_rate'])(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(output_shape)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_tcn_model(input_shape, output_shape, params=None):
    """Create a Temporal Convolutional Network with configurable hyperparameters."""
    if params is None:
        params = {
            'num_filters': 64,
            'num_blocks': 3,
            'dropout_rate': 0.2
        }
    
    def tcn_block(x, filters, dilation_rate):
        shortcut = x
        x = tf.keras.layers.Conv1D(filters, 3, padding='causal', dilation_rate=dilation_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(filters, 3, padding='causal', dilation_rate=dilation_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(params['num_filters'], 3, padding='causal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    for i in range(params['num_blocks']):
        x = tcn_block(x, params['num_filters'], dilation_rate=2**i)
        x = tf.keras.layers.Dropout(params['dropout_rate'])(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(output_shape)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def compile_model(model):
    """Compile the model with appropriate optimizer and loss function."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    return model 