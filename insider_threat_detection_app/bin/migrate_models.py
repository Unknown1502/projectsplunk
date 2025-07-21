#!/usr/bin/env python3
"""
Migration script to move existing models to the Splunk app structure
"""

import os
import sys
import shutil
import pickle
import json
from datetime import datetime

def migrate_models():
    """Migrate existing models to the new app structure."""
    
    print("=== Insider Threat Detection Model Migration ===")
    print(f"Starting migration at {datetime.now()}")
    
    # Define paths
    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(app_root, 'bin', 'models')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models directory: {models_dir}")
    
    # Look for existing models in common locations
    search_paths = [
        os.path.join(app_root, '..', '..', 'insider_threat_detection', 'r1', 'checkpoints'),
        os.path.join(app_root, '..', '..', 'insider_threat_detection', 'models'),
        os.path.join(app_root, '..', '..', 'r1', 'checkpoints'),
        '/opt/ml/models',
        'C:\\ml\\models',
    ]
    
    models_found = []
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"\nSearching in: {search_path}")
            
            # Look for model files
            for file in os.listdir(search_path):
                if file.endswith('.h5') or file.endswith('.keras'):
                    source_path = os.path.join(search_path, file)
                    dest_path = os.path.join(models_dir, file)
                    
                    print(f"Found model: {file}")
                    
                    # Copy model file
                    try:
                        shutil.copy2(source_path, dest_path)
                        print(f"  ✓ Copied to: {dest_path}")
                        models_found.append(file)
                        
                        # Look for associated files
                        base_name = os.path.splitext(file)[0]
                        
                        # Copy scaler
                        scaler_files = [
                            f"{base_name}_scaler.pkl",
                            "scaler.pkl",
                            f"{base_name}.scaler"
                        ]
                        for scaler_file in scaler_files:
                            scaler_path = os.path.join(search_path, scaler_file)
                            if os.path.exists(scaler_path):
                                dest_scaler = os.path.join(models_dir, 'scaler.pkl')
                                shutil.copy2(scaler_path, dest_scaler)
                                print(f"  ✓ Copied scaler: {scaler_file}")
                                break
                        
                        # Copy encoders
                        encoder_files = [
                            f"{base_name}_encoders.pkl",
                            "label_encoders.pkl",
                            f"{base_name}.encoders"
                        ]
                        for encoder_file in encoder_files:
                            encoder_path = os.path.join(search_path, encoder_file)
                            if os.path.exists(encoder_path):
                                dest_encoder = os.path.join(models_dir, 'label_encoders.pkl')
                                shutil.copy2(encoder_path, dest_encoder)
                                print(f"  ✓ Copied encoders: {encoder_file}")
                                break
                        
                        # Copy feature columns
                        feature_files = [
                            f"{base_name}_features.pkl",
                            "feature_columns.pkl",
                            f"{base_name}.features"
                        ]
                        for feature_file in feature_files:
                            feature_path = os.path.join(search_path, feature_file)
                            if os.path.exists(feature_path):
                                dest_features = os.path.join(models_dir, 'feature_columns.pkl')
                                shutil.copy2(feature_path, dest_features)
                                print(f"  ✓ Copied features: {feature_file}")
                                break
                                
                    except Exception as e:
                        print(f"  ✗ Error copying {file}: {e}")
    
    # Create a symlink for 'latest' model
    if models_found:
        latest_model = sorted(models_found)[-1]  # Get the most recent by name
        latest_link = os.path.join(models_dir, 'latest_model.h5')
        
        try:
            if os.path.exists(latest_link):
                os.remove(latest_link)
            
            # On Windows, copy instead of symlink
            if sys.platform == 'win32':
                shutil.copy2(os.path.join(models_dir, latest_model), latest_link)
            else:
                os.symlink(latest_model, latest_link)
            
            print(f"\n✓ Set latest model: {latest_model}")
        except Exception as e:
            print(f"\n✗ Could not set latest model: {e}")
    
    # Create model metadata
    metadata = {
        'migration_date': datetime.now().isoformat(),
        'models_migrated': models_found,
        'models_directory': models_dir,
        'total_models': len(models_found)
    }
    
    metadata_path = os.path.join(models_dir, 'migration_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n=== Migration Summary ===")
    print(f"Models migrated: {len(models_found)}")
    print(f"Location: {models_dir}")
    
    if not models_found:
        print("\n⚠ No models found to migrate!")
        print("Please manually copy your model files to:")
        print(f"  {models_dir}")
        print("\nExpected files:")
        print("  - model.h5 (or .keras)")
        print("  - scaler.pkl")
        print("  - label_encoders.pkl")
        print("  - feature_columns.pkl")
    
    return len(models_found) > 0

def migrate_data():
    """Migrate training data to lookups directory."""
    
    print("\n=== Data Migration ===")
    
    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lookups_dir = os.path.join(app_root, 'lookups')
    
    # Look for CSV files
    search_paths = [
        os.path.join(app_root, '..', '..', 'insider_threat_detection', 'r1'),
        os.path.join(app_root, '..', '..', 'r1'),
        os.path.join(app_root, '..', '..', 'insider_threat_detection', 'data'),
    ]
    
    data_files = ['device.csv', 'http.csv', 'logon.csv', 'email.csv', 'file.csv']
    files_found = []
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"\nSearching in: {search_path}")
            
            for data_file in data_files:
                source_path = os.path.join(search_path, data_file)
                if os.path.exists(source_path):
                    dest_path = os.path.join(lookups_dir, f"training_{data_file}")
                    
                    try:
                        shutil.copy2(source_path, dest_path)
                        print(f"  ✓ Copied {data_file}")
                        files_found.append(data_file)
                    except Exception as e:
                        print(f"  ✗ Error copying {data_file}: {e}")
    
    print(f"\nData files migrated: {len(files_found)}")
    
    return len(files_found) > 0

def create_sample_data():
    """Create sample data for testing."""
    
    print("\n=== Creating Sample Data ===")
    
    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lookups_dir = os.path.join(app_root, 'lookups')
    
    # Create sample insider threat data
    sample_data = """date,user,pc,activity_type,details,threat_label
2024-01-15 09:00:00,john.doe,PC-001,login,Normal login,0
2024-01-15 10:30:00,john.doe,PC-001,file_download,Downloaded project.doc,0
2024-01-15 14:00:00,jane.smith,PC-002,bulk_download,Downloaded 50 files,1
2024-01-15 15:30:00,jane.smith,PC-002,usb_connect,Connected USB device,1
2024-01-15 22:00:00,admin.user,PC-003,admin_action,Modified permissions,0
2024-01-16 02:00:00,service.account,SERVER-01,data_export,Exported database,1
2024-01-16 09:15:00,bob.johnson,PC-004,failed_login,3 failed attempts,0
2024-01-16 10:00:00,alice.williams,PC-005,email_external,Sent to competitor.com,1
"""
    
    sample_path = os.path.join(lookups_dir, 'sample_threat_data.csv')
    
    try:
        with open(sample_path, 'w') as f:
            f.write(sample_data)
        print(f"✓ Created sample data: {sample_path}")
        return True
    except Exception as e:
        print(f"✗ Error creating sample data: {e}")
        return False

def main():
    """Main migration function."""
    
    print("=" * 50)
    print("Insider Threat Detection - Model & Data Migration")
    print("=" * 50)
    
    # Check if running in Splunk context
    if 'SPLUNK_HOME' in os.environ:
        print(f"Splunk Home: {os.environ['SPLUNK_HOME']}")
    else:
        print("⚠ Not running in Splunk context")
    
    # Run migrations
    model_success = migrate_models()
    data_success = migrate_data()
    sample_success = create_sample_data()
    
    print("\n" + "=" * 50)
    print("Migration Complete!")
    print("=" * 50)
    
    if model_success and data_success:
        print("\n✓ All migrations successful!")
        print("\nNext steps:")
        print("1. Restart Splunk")
        print("2. Test the commands:")
        print("   | inputlookup sample_threat_data.csv | insider_threat_predict")
        print("3. Check the dashboard")
    else:
        print("\n⚠ Some migrations failed. Please check the output above.")
        print("\nManual steps may be required:")
        print("1. Copy model files to: bin/models/")
        print("2. Copy data files to: lookups/")
    
    return 0 if (model_success and data_success) else 1

if __name__ == "__main__":
    sys.exit(main())
