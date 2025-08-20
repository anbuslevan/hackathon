import pandas as pd
import polars as pl
import joblib
import json
import warnings
from datetime import datetime
import gc
import logging

from kit import ModelTrainer, TelemetryFeatureEngineer
from data_generator import BASE_DIR, DATA_FILE_NAME

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_MODEL_DIR="model"

def main():
    """Main training pipeline"""
    logger.info("Starting Device Failure Prediction ML Pipeline...")
    
    # Load data efficiently with Polars - streaming mode for large files
    logger.info("Loading data...")
    try:
        # For very large files, use lazy loading and sample for training
        df_lazy = pl.scan_csv(f'{BASE_DIR}/{DATA_FILE_NAME}')
        
        # Sample data for training (adjust sample_size based on your memory)
        sample_size = 1_000_000  # 1M records for training
        total_rows = df_lazy.select(pl.len()).collect().item()
        logger.info(f"Total data size: {total_rows:,} rows")
        
        if total_rows > sample_size:
            logger.info(f"Sampling {sample_size:,} rows for training...")
            # Stratified sampling to maintain class distribution
            df = df_lazy.sample(n=sample_size, seed=42).collect()
        else:
            df = df_lazy.collect()
            
        logger.info(f"Training data shape: {df.shape}")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Fallback to chunked processing
        logger.info("Falling back to chunked processing...")
        df = pl.read_csv("data.csv", n_rows=1_000_000)  # Load first 1M rows
        logger.info(f"Loaded chunk: {df.shape}")
    
    # Initialize feature engineer
    feature_engineer = TelemetryFeatureEngineer()
    
    # Apply feature engineering
    logger.info("Applying feature engineering...")
    df_features = feature_engineer.fit_transform(df)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    X, y = trainer.prepare_data(df_features, feature_engineer.feature_columns, "failure_type")
    
    # Train all models
    results = trainer.train_all_models(X, y)
    
    # Save best model and artifacts
    logger.info("Saving model artifacts...")
    
    # Save best model
    joblib.dump(trainer.best_model, f"{BASE_MODEL_DIR}/model.joblib")
    
    # Save preprocessing pipeline
    preprocessing_artifacts = {
        'feature_engineer': feature_engineer,
        'scaler': trainer.scaler,
        'label_encoder': trainer.label_encoder,
        'feature_columns': feature_engineer.feature_columns,
        'thresholds': trainer.thresholds
    }
    joblib.dump(preprocessing_artifacts, f"{BASE_MODEL_DIR}preprocessing_pipeline.joblib")
    
    # Save metrics report
    final_report = {
        'best_model': trainer.best_model_name,
        'best_model_metrics': results[trainer.best_model_name],
        'all_models_comparison': results,
        'thresholds': trainer.thresholds,
        'feature_count': len(feature_engineer.feature_columns),
        'training_timestamp': datetime.now().isoformat()
    }
    
    with open(f'{BASE_MODEL_DIR}/model_evaluation_report.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    # Feature importance for best model
    if hasattr(trainer.best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_engineer.feature_columns,
            'importance': trainer.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(f'{BASE_MODEL_DIR}/feature_importance.csv', index=False)
        logger.info(f"Feature importance saved to {BASE_MODEL_DIR}/feature_importance.csv")
        
        # Display top 20 features
        print("\nTop 20 Most Important Features:")
        print(feature_importance.head(20).to_string(index=False))
    
    # Memory cleanup
    del df, df_features, X, y
    gc.collect()
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Best model ({trainer.best_model_name}) saved to: {BASE_MODEL_DIR}/model.joblib")
    logger.info(f"Preprocessing pipeline saved to: {BASE_MODEL_DIR}/preprocessing_pipeline.joblib")
    logger.info(f"Evaluation report saved to: {BASE_MODEL_DIR}/model_evaluation_report.json")
    
    return trainer.best_model, preprocessing_artifacts, final_report

if __name__ == "__main__":
    # Run the main training pipeline
    best_model, preprocessing_artifacts, final_report = main()
    