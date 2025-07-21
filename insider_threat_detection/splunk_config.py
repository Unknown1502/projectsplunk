"""
Splunk Configuration Management
This file manages Splunk Enterprise connection details and credentials.
"""

import os
import json
from pathlib import Path

class SplunkConfig:
    """Manage Splunk Enterprise configuration and credentials."""
    
    def __init__(self):
        self.config_file = "splunk_credentials.json"
        self.config = self.load_config()
    
    def load_config(self):
        """Load Splunk configuration from file or create default."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.get_default_config()
        else:
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default Splunk configuration."""
        return {
            "splunk_host": "localhost",
            "splunk_port": 8089,
            "splunk_username": "admin",
            "splunk_password": "",
            "splunk_app": "insider_threat_detection",
            "splunk_index": "main",
            "splunk_sourcetype": "insider_threat",
            "management_port": 8089,
            "web_port": 8000,
            "hec_token": "2be9538b-ac19-41fa-909c-8e06f306805d",
            "hec_port": 8088,
            "ssl_verify": False,
            "deployment_server": "",
            "license_key": "",
            "app_deployment_path": "C:\\Program Files\\Splunk\\etc\\apps\\",
            "model_deployment_path": "C:\\Program Files\\Splunk\\etc\\apps\\insider_threat_detection\\models\\",
            "splunk_home": "C:\\Program Files\\Splunk",
            "splunk_server_name": "Pritesh",
            "default_host": "PRITESH",
            "app_server_port": 8065,
            "index_path": "C:\\Program Files\\Splunk\\var\\lib\\splunk",
            "project_path": "C:\Program Files\Splunk\etc\apps\projectsplunk"
        }
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def setup_interactive(self):
        """Interactive setup of Splunk configuration."""
        print("=" * 60)
        print("SPLUNK ENTERPRISE CONFIGURATION SETUP")
        print("=" * 60)
        
        print("\n1. Basic Connection Settings:")
        self.config["splunk_host"] = input(f"Splunk Host [{self.config['splunk_host']}]: ") or self.config["splunk_host"]
        self.config["splunk_port"] = int(input(f"Splunk Management Port [{self.config['splunk_port']}]: ") or self.config["splunk_port"])
        self.config["web_port"] = int(input(f"Splunk Web Port [{self.config['web_port']}]: ") or self.config["web_port"])
        
        print("\n2. Authentication:")
        self.config["splunk_username"] = input(f"Splunk Username [{self.config['splunk_username']}]: ") or self.config["splunk_username"]
        self.config["splunk_password"] = input(f"Splunk Password: ") or self.config["splunk_password"]
        
        print("\n3. Data Configuration:")
        self.config["splunk_index"] = input(f"Target Index [{self.config['splunk_index']}]: ") or self.config["splunk_index"]
        self.config["splunk_sourcetype"] = input(f"Sourcetype [{self.config['splunk_sourcetype']}]: ") or self.config["splunk_sourcetype"]
        
        print("\n4. HTTP Event Collector (Optional):")
        hec_token = input(f"HEC Token (optional): ")
        if hec_token:
            self.config["hec_token"] = hec_token
            self.config["hec_port"] = int(input(f"HEC Port [{self.config['hec_port']}]: ") or self.config["hec_port"])
        
        print("\n5. Deployment Paths:")
        self.config["app_deployment_path"] = input(f"App Deployment Path [{self.config['app_deployment_path']}]: ") or self.config["app_deployment_path"]
        self.config["model_deployment_path"] = input(f"Model Deployment Path [{self.config['model_deployment_path']}]: ") or self.config["model_deployment_path"]
        
        # Save configuration
        if self.save_config():
            print("\n✅ Configuration saved successfully!")
        else:
            print("\n❌ Failed to save configuration!")
        
        return self.config
    
    def get_connection_string(self):
        """Get Splunk connection string."""
        return f"https://{self.config['splunk_host']}:{self.config['splunk_port']}"
    
    def get_web_url(self):
        """Get Splunk web interface URL."""
        return f"http://{self.config['splunk_host']}:{self.config['web_port']}"
    
    def validate_config(self):
        """Validate current configuration."""
        required_fields = ["splunk_host", "splunk_username", "splunk_password"]
        missing_fields = []
        
        for field in required_fields:
            if not self.config.get(field):
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required configuration: {', '.join(missing_fields)}")
            return False
        
        print("✅ Configuration validation passed!")
        return True
    
    def print_config(self):
        """Print current configuration (hiding sensitive data)."""
        print("\n" + "=" * 50)
        print("CURRENT SPLUNK CONFIGURATION")
        print("=" * 50)
        
        safe_config = self.config.copy()
        # Hide sensitive information
        if safe_config.get("splunk_password"):
            safe_config["splunk_password"] = "*" * len(safe_config["splunk_password"])
        if safe_config.get("hec_token"):
            safe_config["hec_token"] = safe_config["hec_token"][:8] + "..." if len(safe_config["hec_token"]) > 8 else "***"
        
        for key, value in safe_config.items():
            print(f"{key:25}: {value}")
        print("=" * 50)

if __name__ == "__main__":
    config = SplunkConfig()
    config.setup_interactive()
    config.print_config()
