import pandas as pd
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (
    classification_report, precision_recall_curve, auc, f1_score, precision_score, recall_score
)
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import joblib
import warnings
from datetime import datetime
import gc
from typing import Dict, List, Tuple
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TelemetryFeatureEngineer:
    """
    Advanced feature engineering for telemetry data with user-wise, device-wise, 
    and device-model-wise historical patterns.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
    def create_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create time-based features"""
        logger.info("Creating temporal features...")
        
        # df = df.with_columns([
        #     pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S").alias("datetime"),
        #     pl.col("device_production_date").str.strptime(pl.Date, "%Y-%m-%d").alias("prod_date")
        # ])

        df = df.with_columns([
            pl.col("timestamp")
            .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S.%f", strict=False)
            .alias("datetime"),
            pl.col("device_production_date")
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            .alias("prod_date")
        ])
        
        df = df.with_columns([
            pl.col("datetime").dt.hour().alias("hour_of_day"),
            pl.col("datetime").dt.weekday().alias("day_of_week"),
            pl.col("datetime").dt.month().alias("month"),
            (pl.col("datetime").dt.date() - pl.col("prod_date")).dt.total_days().alias("device_age_actual")
        ])
        
        return df
    
    def create_user_historical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create user-wise historical aggregations - memory efficient"""
        logger.info("Creating user historical features...")
        
        # Process in chunks to avoid memory issues
        try:
            # User-level aggregations with memory-efficient operations
            user_features = df.group_by("user_id").agg([
                # Battery patterns - use fewer aggregations
                pl.col("battery_level_percent").mean().alias("user_avg_battery_level"),
                pl.col("battery_temperature_c").mean().alias("user_avg_battery_temp"),
                
                # CPU patterns  
                pl.col("cpu_usage_percent").mean().alias("user_avg_cpu_usage"),
                pl.col("cpu_temperature_c_avg").mean().alias("user_avg_cpu_temp"),
                
                # Memory patterns
                pl.col("memory_used_mb").mean().alias("user_avg_memory_usage"),
                
                # Failure history
                pl.col("failure_occurred").sum().alias("user_total_failures"),
                pl.col("failure_occurred").mean().alias("user_failure_rate"),
                
                # Device usage patterns
                pl.col("record_id").count().alias("user_telemetry_count")
            ])
            
            df = df.join(user_features, on="user_id", how="left")
            
            # Clean up intermediate results
            del user_features
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error in user features: {e}, using simplified version")
            # Fallback to basic user features
            df = df.with_columns([
                pl.lit(0.0).alias("user_avg_battery_level"),
                pl.lit(0.0).alias("user_avg_battery_temp"),
                pl.lit(0.0).alias("user_avg_cpu_usage"),
                pl.lit(0.0).alias("user_avg_cpu_temp"),
                pl.lit(0.0).alias("user_avg_memory_usage"),
                pl.lit(0).alias("user_total_failures"),
                pl.lit(0.0).alias("user_failure_rate"),
                pl.lit(1).alias("user_telemetry_count")
            ])
        
        return df
    
    def create_device_historical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create device-wise historical patterns - memory efficient"""
        logger.info("Creating device historical features...")
        
        try:
            # Simplified device-level aggregations to reduce memory usage
            device_features = df.group_by("device_id").agg([
                # Battery degradation patterns
                pl.col("battery_level_percent").std().alias("device_battery_volatility"),
                pl.col("battery_sudden_shutdowns").sum().alias("device_total_shutdowns"),
                
                # Thermal patterns
                pl.col("thermal_hotspot_temp_c").mean().alias("device_avg_hotspot_temp"),
                pl.col("thermal_hotspot_temp_c").max().alias("device_max_hotspot_temp"),
                
                # Error accumulation - simplified
                pl.col("storage_read_errors").sum().alias("device_total_storage_errors"),
                pl.col("kernel_panics").sum().alias("device_total_kernel_panics"),
                
                # Record count
                pl.col("record_id").count().alias("device_record_count")
            ])
            
            df = df.join(device_features, on="device_id", how="left")
            del device_features
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error in device features: {e}, using simplified version")
            # Fallback
            df = df.with_columns([
                pl.lit(0.0).alias("device_battery_volatility"),
                pl.lit(0).alias("device_total_shutdowns"),
                pl.lit(35.0).alias("device_avg_hotspot_temp"),
                pl.lit(45.0).alias("device_max_hotspot_temp"),
                pl.lit(0).alias("device_total_storage_errors"),
                pl.lit(0).alias("device_total_kernel_panics"),
                pl.lit(1).alias("device_record_count")
            ])
        
        return df
    
    def create_device_model_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create device-model-wise patterns for systemic failure detection"""
        logger.info("Creating device model features...")
        
        model_features = df.group_by("device_model").agg([
            # Model reliability metrics
            pl.col("failure_occurred").mean().alias("model_failure_rate"),
            pl.col("battery_charge_cycles").mean().alias("model_avg_charge_cycles"),
            pl.col("device_age_days").mean().alias("model_avg_device_age"),
            
            # Model-specific thermal characteristics
            pl.col("cpu_temperature_c_avg").mean().alias("model_avg_cpu_temp"),
            pl.col("thermal_hotspot_temp_c").mean().alias("model_avg_hotspot_temp"),
            
            # Model-specific performance baselines
            pl.col("cpu_usage_percent").quantile(0.95).alias("model_cpu_95_percentile"),
            pl.col("memory_used_mb").quantile(0.95).alias("model_memory_95_percentile"),
            
            # Batch-related patterns
            pl.col("device_batch_number").n_unique().alias("model_batch_diversity"),
            pl.col("record_id").count().alias("model_sample_count")
        ])
        
        df = df.join(model_features, on="device_model", how="left")
        return df
    
    def create_anomaly_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create anomaly detection features"""
        logger.info("Creating anomaly features...")
        
        df = df.with_columns([
            # Temperature anomalies
            (pl.col("cpu_temperature_c_avg") > pl.col("model_avg_cpu_temp") * 1.2).alias("cpu_temp_anomaly"),
            (pl.col("thermal_hotspot_temp_c") > pl.col("model_avg_hotspot_temp") * 1.2).alias("hotspot_temp_anomaly"),
            
            # Performance anomalies  
            (pl.col("cpu_usage_percent") > pl.col("model_cpu_95_percentile")).alias("cpu_usage_anomaly"),
            (pl.col("memory_used_mb") > pl.col("model_memory_95_percentile")).alias("memory_usage_anomaly"),
            
            # Battery anomalies
            (pl.col("battery_temperature_c") > 45.0).alias("battery_overheat"),
            (pl.col("battery_level_percent") < 5.0).alias("battery_critical_low"),
            
            # Error rate anomalies
            (pl.col("storage_read_errors") + pl.col("storage_write_errors") > 10).alias("storage_error_spike"),
            (pl.col("kernel_panics") > 0).alias("kernel_panic_occurred")
        ])
        
        return df
    
    def encode_categorical_features(self, df: pl.DataFrame, fit_mode: bool = True) -> pl.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical features...")
        
        categorical_cols = [
            "device_manufacturer", "device_model", "os_name", "os_version",
            "battery_health_status", "battery_charging_status", "warranty_status",
            "user_region", "user_behavior_profile", "kernel_last_boot_reason"
        ]
        
        # Convert to pandas for encoding
        df_pandas = df.to_pandas()
        
        for col in categorical_cols:
            if col in df_pandas.columns:
                if fit_mode:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df_pandas[f"{col}_encoded"] = self.encoders[col].fit_transform(
                        df_pandas[col].astype(str).fillna("unknown")
                    )
                else:
                    if col in self.encoders:
                        # Handle unseen categories
                        unique_vals = df_pandas[col].astype(str).fillna("unknown")
                        encoded_vals = []
                        for val in unique_vals:
                            if val in self.encoders[col].classes_:
                                encoded_vals.append(self.encoders[col].transform([val])[0])
                            else:
                                encoded_vals.append(-1)  # Unknown category
                        df_pandas[f"{col}_encoded"] = encoded_vals
        
        return pl.from_pandas(df_pandas)
    
    def create_interaction_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create interaction features between important variables"""
        logger.info("Creating interaction features...")
        
        df = df.with_columns([
            # Temperature × Usage interactions
            (pl.col("cpu_temperature_c_avg") * pl.col("cpu_usage_percent")).alias("cpu_temp_usage_interaction"),
            (pl.col("battery_temperature_c") * pl.col("battery_level_percent")).alias("battery_temp_level_interaction"),
            
            # Age × Performance interactions
            (pl.col("device_age_days") * pl.col("cpu_usage_percent")).alias("age_cpu_interaction"),
            (pl.col("device_age_days") * pl.col("battery_level_percent")).alias("age_battery_interaction"),
            
            # Error rate combinations
            (pl.col("storage_read_errors") + pl.col("storage_write_errors") + pl.col("memory_ecc_corrected_errors")).alias("total_hardware_errors"),
            (pl.col("software_system_crashes") + pl.col("software_app_crashes") + pl.col("kernel_panics")).alias("total_software_errors")
        ])
        
        return df
    
    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        
        # Apply all feature engineering steps
        df = self.create_temporal_features(df)
        df = self.create_user_historical_features(df)
        df = self.create_device_historical_features(df)
        df = self.create_device_model_features(df)
        df = self.create_anomaly_features(df)
        df = self.encode_categorical_features(df, fit_mode=True)
        df = self.create_interaction_features(df)
        
        # Get feature columns (exclude identifiers, target, and non-encoded categoricals)
        exclude_cols = [
            "record_id", "device_id", "user_id", "timestamp", "datetime", "prod_date",
            "device_production_date", "failure_timestamp", "device_manufacturer",
            "device_model", "os_name", "os_version", "battery_health_status", 
            "battery_charging_status", "warranty_status", "user_region", 
            "user_behavior_profile", "kernel_last_boot_reason", "device_batch_number",
            "software_firmware_version", "software_security_patch", "os_build_number",
            "cpu_core_frequencies_mhz", "failure_type", "failure_occurred"
        ]
        
        # Only include numeric columns and encoded categorical columns
        self.feature_columns = []
        for col in df.columns:
            if col not in exclude_cols:
                # Check if it's a numeric column or an encoded categorical
                if col.endswith('_encoded') or df[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.Boolean]:
                    self.feature_columns.append(col)
        logger.info(f"Total features created: {len(self.feature_columns)}")
        
        return df
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform new data using fitted encoders"""
        logger.info("Transforming new data...")
        
        df = self.create_temporal_features(df)
        df = self.create_user_historical_features(df)
        df = self.create_device_historical_features(df)  
        df = self.create_device_model_features(df)
        df = self.create_anomaly_features(df)
        df = self.encode_categorical_features(df, fit_mode=False)
        df = self.create_interaction_features(df)
        
        return df

class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for detecting systemic failures across related devices
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 8):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class ModelTrainer:
    """
    Production-grade model training with multiple algorithms and proper evaluation
    """
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.thresholds = {}
        self.scaler = RobustScaler()
        
    def prepare_data(self, df: pl.DataFrame, feature_cols: List[str], target_col: str = "failure_type") -> Tuple:
        """Prepare data for training - memory efficient"""
        logger.info("Preparing data for training...")
        
        # Convert to pandas in chunks to manage memory
        logger.info(f"Converting to pandas with {len(feature_cols)} features...")
        
        try:
            # Convert only necessary columns
            necessary_cols = feature_cols + [target_col]
            df_subset = df.select(necessary_cols)
            df_pandas = df_subset.to_pandas()
            
            # Clean up
            del df_subset
            gc.collect()
            
            # Handle missing values
            logger.info("Handling missing values...")
            df_pandas[feature_cols] = df_pandas[feature_cols].fillna(0)
            
            # Prepare features and target
            X = df_pandas[feature_cols].values.astype(np.float32)  # Use float32 to save memory
            
            # Encode target labels
            if target_col == "failure_type":
                # Multi-class classification
                le = LabelEncoder()
                y = le.fit_transform(df_pandas[target_col].fillna("none"))
                self.label_encoder = le
            else:
                # Binary classification
                y = df_pandas[target_col].astype(int).values
                
            # Clean up pandas dataframe
            del df_pandas
            gc.collect()
            
            # Scale features in chunks if data is too large
            logger.info("Scaling features...")
            if X.shape[0] > 500_000:  # If more than 500k samples, use partial fit
                logger.info("Using incremental scaling for large dataset...")
                self.scaler.fit(X[:100_000])  # Fit on first 100k samples
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = self.scaler.fit_transform(X)
            
            logger.info(f"Data shape: {X_scaled.shape}")
            logger.info(f"Class distribution: {np.bincount(y)}")
            
            return X_scaled, y
            
        except MemoryError:
            logger.error("Memory error during data preparation. Try reducing sample size.")
            raise
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise
    
    def train_xgboost(self, X_train, X_val, y_train, y_val) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        logger.info("Training XGBoost...")
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.9,
            n_estimators=1000,
            early_stopping_rounds=50,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return model
    
    def create_device_graph(self, df: pl.DataFrame, feature_cols: List[str]) -> Data:
        """Create graph structure for GNN based on device relationships"""
        logger.info("Creating device graph for GNN...")
        
        df_pandas = df.to_pandas()
        
        # Create edges based on device model, batch, and temporal proximity
        edges = []
        node_features = []
        node_labels = []
        
        # Group by device_model and batch for creating edges
        device_groups = df_pandas.groupby(['device_model', 'device_batch_number'])
        
        node_id = 0
        device_to_node = {}
        
        for (model, batch), group in device_groups:
            if len(group) < 2:
                continue
                
            devices = group['device_id'].unique()
            
            # Create features and labels for each device
            for device in devices:
                device_data = group[group['device_id'] == device].iloc[-1]  # Latest record
                
                device_to_node[device] = node_id
                node_features.append(device_data[feature_cols].values)
                
                # Use failure_type as label
                if device_data['failure_type'] == 'none' or pd.isna(device_data['failure_type']):
                    node_labels.append(0)
                else:
                    node_labels.append(self.label_encoder.transform([device_data['failure_type']])[0])
                
                node_id += 1
            
            # Create edges between devices in same model/batch
            device_nodes = [device_to_node[d] for d in devices]
            for i in range(len(device_nodes)):
                for j in range(i + 1, len(device_nodes)):
                    edges.append([device_nodes[i], device_nodes[j]])
                    edges.append([device_nodes[j], device_nodes[i]])  # Undirected
        
        if len(edges) == 0 or len(node_features) == 0:
            logger.warning("No valid graph structure created")
            return None
            
        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(np.array(node_features), dtype=torch.float32)
        y = torch.tensor(node_labels, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def train_gnn(self, graph_data: Data) -> GraphNeuralNetwork:
        """Train Graph Neural Network"""
        logger.info("Training Graph Neural Network...")
        
        if graph_data is None:
            logger.error("No graph data available for GNN training")
            return None
        
        model = GraphNeuralNetwork(
            input_dim=graph_data.x.shape[1],
            hidden_dim=128,
            num_classes=len(np.unique(graph_data.y.numpy()))
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.NLLLoss()
        
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(graph_data.x, graph_data.edge_index)
            loss = criterion(out, graph_data.y)
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                logger.info(f'GNN Epoch {epoch}, Loss: {loss.item():.4f}')
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating {model_name}...")
        
        if model_name == 'gnn':
            # Special handling for GNN
            model.eval()
            with torch.no_grad():
                out = model(X_test.x, X_test.edge_index)
                y_pred = out.argmax(dim=1).numpy()
                y_test_np = X_test.y.numpy()
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            y_test_np = y_test
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y_test_np).mean(),
            'f1_macro': f1_score(y_test_np, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test_np, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test_np, y_pred, average='macro'),
            'recall_macro': recall_score(y_test_np, y_pred, average='macro')
        }
        
        if model_name != 'gnn':
            # Calculate PR-AUC for each class
            pr_aucs = []
            for class_idx in range(len(np.unique(y_test))):
                y_true_binary = (y_test == class_idx).astype(int)
                y_score = y_pred_proba[:, class_idx]
                precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
                pr_auc = auc(recall, precision)
                pr_aucs.append(pr_auc)
            
            metrics['pr_auc_macro'] = np.mean(pr_aucs)
        
        # Classification report
        class_report = classification_report(y_test_np, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        logger.info(f"{model_name} - F1 Macro: {metrics['f1_macro']:.4f}, PR-AUC: {metrics.get('pr_auc_macro', 'N/A')}")
        
        return metrics
    
    def optimize_thresholds(self, model, X_val, y_val) -> Dict:
        """Optimize classification thresholds for each class"""
        logger.info("Optimizing classification thresholds...")
        
        y_pred_proba = model.predict_proba(X_val)
        thresholds = {}
        
        for class_idx in range(y_pred_proba.shape[1]):
            y_true_binary = (y_val == class_idx).astype(int)
            y_scores = y_pred_proba[:, class_idx]
            
            # Find optimal threshold using F1 score
            precision, recall, threshold_vals = precision_recall_curve(y_true_binary, y_scores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = threshold_vals[optimal_idx] if optimal_idx < len(threshold_vals) else 0.5
            
            thresholds[f'class_{class_idx}'] = optimal_threshold
        
        return thresholds
    
    def train_all_models(self, X, y) -> Dict:
        """Train all models and return results"""
        logger.info("Starting comprehensive model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        models_results = {}
        
        xgb_model = self.train_xgboost(X_train, X_val, y_train, y_val)
        self.models['xgboost'] = xgb_model
        models_results['xgboost'] = self.evaluate_model(xgb_model, X_test, y_test, 'xgboost')
        
        # Select best model based on PR-AUC macro
        best_score = 0
        for model_name, metrics in models_results.items():
            if 'pr_auc_macro' in metrics and metrics['pr_auc_macro'] > best_score:
                best_score = metrics['pr_auc_macro']
                self.best_model = self.models[model_name]
                self.best_model_name = model_name
        
        # Optimize thresholds for best model
        if self.best_model:
            self.thresholds = self.optimize_thresholds(self.best_model, X_val, y_val)
        
        logger.info(f"Best model: {self.best_model_name} with PR-AUC: {best_score:.4f}")
        
        return models_results

class ProductionPredictor:
    """
    Production-ready predictor class for real-time telemetry failure prediction
    """
    
    def __init__(self, model_path: str, preprocessing_path: str):
        """Load trained model and preprocessing pipeline"""
        logger.info("Loading production model...")
        
        self.model = joblib.load(model_path)
        self.preprocessing_artifacts = joblib.load(preprocessing_path)
        
        self.feature_engineer = self.preprocessing_artifacts['feature_engineer']
        self.scaler = self.preprocessing_artifacts['scaler']
        self.label_encoder = self.preprocessing_artifacts['label_encoder']
        self.feature_columns = self.preprocessing_artifacts['feature_columns']
        self.thresholds = self.preprocessing_artifacts['thresholds']
        
        logger.info("Production model loaded successfully")
    
    def predict_failure(self, telemetry_data: Dict) -> Dict:
        """
        Predict failure for incoming telemetry data
        
        Args:
            telemetry_data: Dictionary containing telemetry fields as per schema
            
        Returns:
            Dictionary with prediction results and confidence scores
        """
        try:
            # Convert to DataFrame
            df = pl.DataFrame([telemetry_data])
            
            # Note: For production, you'd need historical data for user/device/model features
            # This is a simplified version - in practice you'd query your database
            # to get historical aggregations
            
            # Apply feature engineering (transform mode)
            df_features = self.feature_engineer.transform(df)
            
            # Prepare features
            df_pandas = df_features.to_pandas()
            X = df_pandas[self.feature_columns].fillna(0).values
            X_scaled = self.scaler.transform(X)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get class predictions with optimized thresholds
            predicted_classes = []
            for class_idx, prob in enumerate(probabilities):
                threshold = self.thresholds.get(f'class_{class_idx}', 0.5)
                if prob >= threshold:
                    predicted_classes.append(class_idx)
            
            # If no class meets threshold, use highest probability
            if not predicted_classes:
                predicted_classes = [np.argmax(probabilities)]
            
            # Decode predictions
            failure_types = []
            for class_idx in predicted_classes:
                failure_type = self.label_encoder.inverse_transform([class_idx])[0]
                failure_types.append(failure_type)
            
            # Prepare response
            result = {
                'device_id': telemetry_data.get('device_id', 'unknown'),
                'timestamp': telemetry_data.get('timestamp', datetime.now().isoformat()),
                'failure_predicted': len([ft for ft in failure_types if ft != 'none']) > 0,
                'failure_types': failure_types,
                'failure_probabilities': {
                    self.label_encoder.inverse_transform([i])[0]: float(prob) 
                    for i, prob in enumerate(probabilities)
                },
                'max_confidence': float(max(probabilities)),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'device_id': telemetry_data.get('device_id', 'unknown'),
                'error': str(e),
                'prediction_timestamp': datetime.now().isoformat()
            }
    
    def batch_predict(self, telemetry_batch: List[Dict]) -> List[Dict]:
        """Batch prediction for multiple telemetry records"""
        logger.info(f"Processing batch of {len(telemetry_batch)} records...")
        
        results = []
        for telemetry in telemetry_batch:
            result = self.predict_failure(telemetry)
            results.append(result)
        
        return results

class ModelMonitor:
    """
    Model performance monitoring and drift detection
    """
    
    def __init__(self):
        self.prediction_history = []
        self.performance_metrics = {}
    
    def log_prediction(self, prediction_result: Dict, actual_failure: str = None):
        """Log prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'device_id': prediction_result.get('device_id'),
            'predicted_failure': prediction_result.get('failure_predicted'),
            'predicted_types': prediction_result.get('failure_types'),
            'max_confidence': prediction_result.get('max_confidence'),
            'actual_failure': actual_failure
        }
        
        self.prediction_history.append(log_entry)
    
    def calculate_drift_metrics(self, recent_data: pl.DataFrame) -> Dict:
        """Calculate data drift metrics"""
        # Implement data drift detection logic
        # This would compare feature distributions between training and recent data
        pass
    
    def generate_performance_report(self) -> Dict:
        """Generate model performance report"""
        if not self.prediction_history:
            return {"status": "No predictions logged"}
        
        df = pd.DataFrame(self.prediction_history)
        
        # Calculate metrics where actual labels are available
        labeled_data = df.dropna(subset=['actual_failure'])
        
        if len(labeled_data) > 0:
            accuracy = (labeled_data['predicted_failure'] == 
                       (labeled_data['actual_failure'] != 'none')).mean()
            
            return {
                'total_predictions': len(df),
                'labeled_predictions': len(labeled_data),
                'accuracy': float(accuracy),
                'average_confidence': df['max_confidence'].mean(),
                'failure_prediction_rate': df['predicted_failure'].mean(),
                'report_timestamp': datetime.now().isoformat()
            }
        
        return {
            'total_predictions': len(df),
            'labeled_predictions': 0,
            'average_confidence': float(df['max_confidence'].mean()),
            'failure_prediction_rate': float(df['predicted_failure'].mean()),
            'report_timestamp': datetime.now().isoformat()
        }
