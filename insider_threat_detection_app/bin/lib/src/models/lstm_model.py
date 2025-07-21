"""LSTM/GRU model implementation for insider threat detection."""

import tensorflow as tf
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.regularizers import l1_l2
    from tensorflow.keras.optimizers import Adam
except ImportError:
    # Fallback for different TensorFlow versions
    from keras.models import Sequential
    from keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from keras.regularizers import l1_l2
    from keras.optimizers import Adam

from typing import Tuple, Dict, Any

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.base_model import BaseModel
from config.model_config import MODEL_CONFIG
from config.settings import LEARNING_RATE


class InsiderThreatLSTM(BaseModel):
    """LSTM/GRU model for insider threat detection."""
    
    def __init__(self, use_gru: bool = False):
        super().__init__("InsiderThreatLSTM")
        self.use_gru = use_gru
        self.model_config = MODEL_CONFIG
    
    def build_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Build the LSTM/GRU model architecture."""
        self.logger.info(f"Building {'GRU' if self.use_gru else 'LSTM'} model with input shape: {input_shape}")
        
        model = Sequential()
        
        # Add recurrent layers
        for i, layer_config in enumerate(self.model_config['gru_layers']):
            if i == 0:
                # First layer needs input shape
                if self.use_gru:
                    model.add(GRU(
                        units=layer_config['units'],
                        return_sequences=layer_config['return_sequences'],
                        input_shape=input_shape,
                        dropout=layer_config['dropout'],
                        recurrent_dropout=layer_config['recurrent_dropout'],
                        kernel_regularizer=l1_l2(
                            l1=layer_config['l1_reg'],
                            l2=layer_config['l2_reg']
                        )
                    ))
                else:
                    model.add(LSTM(
                        units=layer_config['units'],
                        return_sequences=layer_config['return_sequences'],
                        input_shape=input_shape,
                        dropout=layer_config['dropout'],
                        recurrent_dropout=layer_config['recurrent_dropout'],
                        kernel_regularizer=l1_l2(
                            l1=layer_config['l1_reg'],
                            l2=layer_config['l2_reg']
                        )
                    ))
            else:
                # Subsequent layers
                if self.use_gru:
                    model.add(GRU(
                        units=layer_config['units'],
                        return_sequences=layer_config['return_sequences'],
                        dropout=layer_config['dropout'],
                        recurrent_dropout=layer_config['recurrent_dropout'],
                        kernel_regularizer=l1_l2(
                            l1=layer_config['l1_reg'],
                            l2=layer_config['l2_reg']
                        )
                    ))
                else:
                    model.add(LSTM(
                        units=layer_config['units'],
                        return_sequences=layer_config['return_sequences'],
                        dropout=layer_config['dropout'],
                        recurrent_dropout=layer_config['recurrent_dropout'],
                        kernel_regularizer=l1_l2(
                            l1=layer_config['l1_reg'],
                            l2=layer_config['l2_reg']
                        )
                    ))
            
            # Add batch normalization after each recurrent layer
            model.add(BatchNormalization())
        
        # Add dense layers
        for layer_config in self.model_config['dense_layers']:
            model.add(Dense(
                units=layer_config['units'],
                activation=layer_config['activation'],
                kernel_regularizer=l1_l2(
                    l1=layer_config['l1_reg'],
                    l2=layer_config['l2_reg']
                )
            ))
            model.add(Dropout(layer_config['dropout']))
        
        # Add output layer
        output_config = self.model_config['output_layer']
        model.add(Dense(
            units=output_config['units'],
            activation=output_config['activation']
        ))
        
        self.model = model
        self.logger.info("Model architecture built successfully")
        
        return model
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration parameters."""
        return {
            "model_type": "GRU" if self.use_gru else "LSTM",
            "architecture": self.model_config,
            "learning_rate": LEARNING_RATE,
            "total_layers": len(self.model_config['gru_layers']) + len(self.model_config['dense_layers']) + 1,
            "regularization": "L1+L2",
            "batch_normalization": True,
            "dropout": True
        }
    
    def compile_model(self, learning_rate: float = LEARNING_RATE, class_weight: dict = None) -> None:
        """Compile the model with Adam optimizer and optional class weights."""
        optimizer = Adam(learning_rate=learning_rate)
        
        # Use proper TensorFlow metric objects instead of strings
        from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
        
        # Store class weights for training
        self.class_weight = class_weight
        
        super().compile_model(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[BinaryAccuracy(name='accuracy'), 
                    Precision(name='precision'), 
                    Recall(name='recall')]
        )
        
        if class_weight:
            self.logger.info(f"Model compiled with class weights: {class_weight}")
        else:
            self.logger.info("Model compiled without class weights")
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get detailed architecture summary."""
        if self.model is None:
            return {"error": "Model not built"}
        
        summary = {
            "model_type": "GRU" if self.use_gru else "LSTM",
            "total_layers": len(self.model.layers),
            "layer_details": self.get_layer_info(),
            "parameter_count": self.count_parameters(),
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape
        }
        
        return summary
    
    def create_bidirectional_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Create a bidirectional version of the model."""
        self.logger.info(f"Building Bidirectional {'GRU' if self.use_gru else 'LSTM'} model")
        
        from tensorflow.keras.layers import Bidirectional
        
        model = Sequential()
        
        # Add bidirectional recurrent layers
        for i, layer_config in enumerate(self.model_config['gru_layers']):
            if self.use_gru:
                recurrent_layer = GRU(
                    units=layer_config['units'],
                    return_sequences=layer_config['return_sequences'],
                    dropout=layer_config['dropout'],
                    recurrent_dropout=layer_config['recurrent_dropout'],
                    kernel_regularizer=l1_l2(
                        l1=layer_config['l1_reg'],
                        l2=layer_config['l2_reg']
                    )
                )
            else:
                recurrent_layer = LSTM(
                    units=layer_config['units'],
                    return_sequences=layer_config['return_sequences'],
                    dropout=layer_config['dropout'],
                    recurrent_dropout=layer_config['recurrent_dropout'],
                    kernel_regularizer=l1_l2(
                        l1=layer_config['l1_reg'],
                        l2=layer_config['l2_reg']
                    )
                )
            
            if i == 0:
                model.add(Bidirectional(recurrent_layer, input_shape=input_shape))
            else:
                model.add(Bidirectional(recurrent_layer))
            
            model.add(BatchNormalization())
        
        # Add dense layers (same as regular model)
        for layer_config in self.model_config['dense_layers']:
            model.add(Dense(
                units=layer_config['units'],
                activation=layer_config['activation'],
                kernel_regularizer=l1_l2(
                    l1=layer_config['l1_reg'],
                    l2=layer_config['l2_reg']
                )
            ))
            model.add(Dropout(layer_config['dropout']))
        
        # Add output layer
        output_config = self.model_config['output_layer']
        model.add(Dense(
            units=output_config['units'],
            activation=output_config['activation']
        ))
        
        self.model = model
        self.logger.info("Bidirectional model architecture built successfully")
        
        return model
    
    def create_attention_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Create a model with attention mechanism."""
        self.logger.info("Building model with attention mechanism")
        
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First GRU/LSTM layer
        if self.use_gru:
            x = GRU(
                units=self.model_config['gru_layers'][0]['units'],
                return_sequences=True,
                dropout=self.model_config['gru_layers'][0]['dropout'],
                recurrent_dropout=self.model_config['gru_layers'][0]['recurrent_dropout']
            )(inputs)
        else:
            x = LSTM(
                units=self.model_config['gru_layers'][0]['units'],
                return_sequences=True,
                dropout=self.model_config['gru_layers'][0]['dropout'],
                recurrent_dropout=self.model_config['gru_layers'][0]['recurrent_dropout']
            )(inputs)
        
        x = BatchNormalization()(x)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=4,
            key_dim=self.model_config['gru_layers'][0]['units'] // 4
        )(x, x)
        
        # Add & Norm
        x = LayerNormalization()(x + attention_output)
        
        # Second GRU/LSTM layer
        if len(self.model_config['gru_layers']) > 1:
            if self.use_gru:
                x = GRU(
                    units=self.model_config['gru_layers'][1]['units'],
                    return_sequences=False,
                    dropout=self.model_config['gru_layers'][1]['dropout'],
                    recurrent_dropout=self.model_config['gru_layers'][1]['recurrent_dropout']
                )(x)
            else:
                x = LSTM(
                    units=self.model_config['gru_layers'][1]['units'],
                    return_sequences=False,
                    dropout=self.model_config['gru_layers'][1]['dropout'],
                    recurrent_dropout=self.model_config['gru_layers'][1]['recurrent_dropout']
                )(x)
            
            x = BatchNormalization()(x)
        
        # Dense layers
        for layer_config in self.model_config['dense_layers']:
            x = Dense(
                units=layer_config['units'],
                activation=layer_config['activation'],
                kernel_regularizer=l1_l2(
                    l1=layer_config['l1_reg'],
                    l2=layer_config['l2_reg']
                )
            )(x)
            x = Dropout(layer_config['dropout'])(x)
        
        # Output layer
        outputs = Dense(
            units=self.model_config['output_layer']['units'],
            activation=self.model_config['output_layer']['activation']
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        self.model = model
        self.logger.info("Attention model architecture built successfully")
        
        return model
