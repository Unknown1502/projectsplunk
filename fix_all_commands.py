"""
Fix all Splunk command Python files to handle import issues
"""
import os
import shutil

SPLUNK_APP_DIR = r"C:\Program Files\Splunk\etc\apps\insider_threat_detection_app"
COMMANDS = ["insider_threat_predict", "insider_threat_train", "insider_threat_monitor", 
            "insider_threat_score", "insider_threat_explain"]

# Common import fix header
IMPORT_FIX_HEADER = '''import sys
import os

# Fix Python path for Splunk app
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
paths_to_add = [
    os.path.join(app_dir, 'bin'),
    os.path.join(app_dir, 'bin', 'lib'),
    os.path.join(app_dir, 'bin', 'lib', 'src'),
    os.path.join(app_dir, 'bin', 'lib', 'config'),
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Set environment variables for config
os.environ['APP_ROOT'] = app_dir
os.environ['MODELS_DIR'] = os.path.join(app_dir, 'bin', 'models')
os.environ['LOOKUPS_DIR'] = os.path.join(app_dir, 'lookups')

'''

def fix_command_file(command_name):
    """Fix a single command file"""
    file_path = os.path.join(SPLUNK_APP_DIR, "bin", f"{command_name}.py")
    backup_path = os.path.join(SPLUNK_APP_DIR, "bin", f"{command_name}_backup.py")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Backup original file
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Backed up: {command_name}.py")
    
    # Read original content
    with open(file_path, 'r') as f:
        original_content = f.read()
    
    # Check if already fixed
    if "Fix Python path for Splunk app" in original_content:
        print(f"Already fixed: {command_name}.py")
        return True
    
    # Find the first import statement
    lines = original_content.split('\n')
    insert_index = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_index = i
            break
    
    # Insert the fix header before the first import
    fixed_lines = lines[:insert_index] + IMPORT_FIX_HEADER.split('\n') + lines[insert_index:]
    fixed_content = '\n'.join(fixed_lines)
    
    # Write fixed content
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed: {command_name}.py")
    return True

def create_init_files():
    """Create __init__.py files in necessary directories"""
    dirs_needing_init = [
        os.path.join(SPLUNK_APP_DIR, "bin"),
        os.path.join(SPLUNK_APP_DIR, "bin", "lib"),
    ]
    
    for dir_path in dirs_needing_init:
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Splunk app package\n")
            print(f"Created: {init_file}")

def update_config_settings():
    """Update config/settings.py to work in Splunk environment"""
    settings_path = os.path.join(SPLUNK_APP_DIR, "bin", "lib", "config", "settings.py")
    
    if os.path.exists(settings_path):
        # Read current settings
        with open(settings_path, 'r') as f:
            content = f.read()
        
        # Add Splunk-specific settings if not already present
        if "SPLUNK_APP_DIR" not in content:
            splunk_settings = '''
# Splunk-specific settings
import os
SPLUNK_APP_DIR = os.environ.get('APP_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
BASE_DIR = SPLUNK_APP_DIR
DATA_DIR = os.path.join(SPLUNK_APP_DIR, 'lookups')
MODEL_DIR = os.path.join(SPLUNK_APP_DIR, 'bin', 'models')
LOG_DIR = os.path.join(SPLUNK_APP_DIR, 'logs')

# Create directories if they don't exist
for dir_path in [MODEL_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)
'''
            # Prepend to settings file
            with open(settings_path, 'w') as f:
                f.write(splunk_settings + '\n' + content)
            print("Updated: config/settings.py")

def main():
    print("Fixing Splunk App Python Commands")
    print("=" * 50)
    
    # Create __init__.py files
    print("\n1. Creating __init__.py files...")
    create_init_files()
    
    # Fix each command
    print("\n2. Fixing command files...")
    for command in COMMANDS:
        fix_command_file(command)
    
    # Update config settings
    print("\n3. Updating config settings...")
    update_config_settings()
    
    print("\n" + "=" * 50)
    print("All fixes applied!")
    print("\nNext steps:")
    print("1. Restart Splunk")
    print("2. Test commands in Splunk Web")

if __name__ == "__main__":
    main()
