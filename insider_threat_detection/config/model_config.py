"""Model architecture and hyperparameter configuration."""

# Model architecture configuration
MODEL_CONFIG = {
    'gru_layers': [
        {
            'units': 24,
            'return_sequences': True,
            'dropout': 0.4,
            'recurrent_dropout': 0.4,
            'l1_reg': 0.01,
            'l2_reg': 0.01
        },
        {
            'units': 12,
            'return_sequences': False,
            'dropout': 0.5,
            'recurrent_dropout': 0.5,
            'l1_reg': 0.01,
            'l2_reg': 0.01
        }
    ],
    'dense_layers': [
        {
            'units': 6,
            'activation': 'relu',
            'dropout': 0.6,
            'l1_reg': 0.01,
            'l2_reg': 0.01
        },
        {
            'units': 3,
            'activation': 'relu',
            'dropout': 0.5,
            'l1_reg': 0.01,
            'l2_reg': 0.01
        }
    ],
    'output_layer': {
        'units': 1,
        'activation': 'sigmoid'
    }
}

# Feature columns configuration
FEATURE_COLUMNS = [
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_weekend', 'is_off_hours', 'activity_type_encoded',
    'user_freq', 'pc_freq', 'details_freq',
    'total_activities', 'avg_hour', 'hour_std', 'unique_pcs',
    'off_hours_ratio', 'weekend_ratio', 'activity_entropy',
    'anomaly_score'
]

# Categorical columns for encoding
CATEGORICAL_COLUMNS = ['user', 'pc', 'activity_type', 'details']
LOW_CARDINALITY_COLUMNS = ['activity_type']

# Time-based features
TIME_FEATURES = {
    'hour_sin': lambda df: np.sin(2 * np.pi * df['hour'] / 24),
    'hour_cos': lambda df: np.cos(2 * np.pi * df['hour'] / 24),
    'day_sin': lambda df: np.sin(2 * np.pi * df['day_of_week'] / 7),
    'day_cos': lambda df: np.cos(2 * np.pi * df['day_of_week'] / 7)
}

# User behavior features
USER_STATS_CONFIG = {
    'aggregations': {
        'id': 'count',
        'hour': ['mean', 'std', 'min', 'max'],
        'is_off_hours': 'mean',
        'is_weekend': 'mean',
        'pc': 'nunique',
        'activity_type': lambda x: x.value_counts().to_dict()
    },
    'column_names': [
        'user', 'total_activities', 'avg_hour', 'hour_std',
        'min_hour', 'max_hour', 'off_hours_ratio',
        'weekend_ratio', 'unique_pcs', 'activity_dist'
    ]
}

# Anomaly detection features
ANOMALY_FEATURES = [
    'hour', 'day_of_week', 'total_activities', 'unique_pcs',
    'off_hours_ratio', 'activity_entropy'
]

# Threat scoring weights
THREAT_SCORING_WEIGHTS = {
    'normalized_features_weight': 0.2,
    'anomaly_score_weight': 0.3,
    'off_hours_weight': 0.1,
    'weekend_weight': 0.05
}
