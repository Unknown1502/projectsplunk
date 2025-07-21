#!/usr/bin/env python3
"""
Update all paths in the project after moving to Splunk apps directory
This script updates all references from the old path to the new path
"""

import os
import json
import re

def update_file_content(file_path, old_patterns, new_path_base):
    """Update file content with new paths"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace various path patterns
        replacements = [
            # Python raw strings
            (r'r"C:\\Users\\nikhi\\projectsplunk\\r1"', f'r"{new_path_base}\\r1"'),
            (r'r"C:\\Users\\nikhi\\projectsplunk"', f'r"{new_path_base}"'),
            
            # JSON/string paths with double backslashes
            (r'C:\\\\Users\\\\nikhi\\\\projectsplunk\\\\insider_threat_detection', new_path_base.replace('\\', '\\\\')),
            (r'C:\\\\Users\\\\nikhi\\\\projectsplunk', new_path_base.replace('\\', '\\\\')),
            
            # Regular string paths
            (r'C:\\Users\\nikhi\\projectsplunk\\insider_threat_detection', new_path_base),
            (r'C:\\Users\\nikhi\\projectsplunk', new_path_base),
            
            # Forward slash paths
            (r'C:/Users/nikhi/projectsplunk/insider_threat_detection', new_path_base.replace('\\', '/')),
            (r'C:/Users/nikhi/projectsplunk', new_path_base.replace('\\', '/')),
            
            # Relative paths to r1 directory (update to absolute)
            (r'"\.\./r1/checkpoints/', f'"{new_path_base}\\r1\\checkpoints\\'),
            (r'"r1/checkpoints/', f'"{new_path_base}\\r1\\checkpoints\\'),
            (r'"r1/', f'"{new_path_base}\\r1\\'),
            (r"'r1/checkpoints/", f"'{new_path_base}\\r1\\checkpoints\\"),
            (r"'r1/", f"'{new_path_base}\\r1\\"),
        ]
        
        for old_pattern, new_pattern in replacements:
            content = content.replace(old_pattern, new_pattern)
        
        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def update_all_paths():
    """Main function to update all paths in the project"""
    
    # New base path
    new_base_path = "C:\\Program Files\\Splunk\\etc\\apps\\projectsplunk"
    
    # Files to update based on search results
    files_to_update = [
        # Python files
        "insider_threat_detection/config/settings.py",
        "insider_threat_detection/splunk_config.py",
        "insider_threat_detection/automated_splunk_integration.py",
        "insider_threat_detection/automated_splunk_integration_fixed.py",
        "insider_threat_detection/predict_realtime.py",
        "insider_threat_detection/direct_file_upload.py",
        "insider_threat_detection/fix_splunk_integration_complete.py",
        
        # JSON files
        "insider_threat_detection/splunk_credentials.json",
        "insider_threat_detection/execution_report_20250720_141406.json",
        "insider_threat_detection/execution_report_20250720_142614.json",
        "insider_threat_detection/execution_report_20250720_161931.json",
        
        # Batch files
        "insider_threat_detection/move_to_splunk.bat"
    ]
    
    print("UPDATING ALL PROJECT PATHS")
    print("=" * 60)
    print(f"New base path: {new_base_path}")
    print()
    
    updated_count = 0
    error_count = 0
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            print(f"Updating: {file_path}")
            if update_file_content(file_path, [], new_base_path):
                print(f"  [OK] Updated successfully")
                updated_count += 1
            else:
                print(f"  - No changes needed")
        else:
            print(f"  [ERROR] File not found: {file_path}")
            error_count += 1
    
    # Special handling for settings.py - update DATA_PATH
    settings_file = "insider_threat_detection/config/settings.py"
    if os.path.exists(settings_file):
        print(f"\nSpecial update for {settings_file}")
        with open(settings_file, 'r') as f:
            content = f.read()
        
        # Update DATA_PATH specifically
        # Use raw string to avoid regex escape issues
        new_data_path = new_base_path.replace('\\', '\\\\')
        content = re.sub(
            r'DATA_PATH = r"[^"]*"',
            f'DATA_PATH = r"{new_data_path}\\\\r1"',
            content
        )
        
        with open(settings_file, 'w') as f:
            f.write(content)
        print("  [OK] Updated DATA_PATH")
    
    print("\n" + "=" * 60)
    print(f"UPDATE COMPLETE!")
    print(f"Files updated: {updated_count}")
    print(f"Errors: {error_count}")
    print("=" * 60)
    
    # Create a test script to verify all paths
    create_test_script(new_base_path)

def create_test_script(base_path):
    """Create a test script to verify all paths are correct"""
    
    test_script = f'''#!/usr/bin/env python3
"""
Test script to verify all paths are correctly updated
"""

import os
import json
import sys

def test_paths():
    """Test all critical paths in the project"""
    
    print("TESTING ALL PROJECT PATHS")
    print("=" * 60)
    
    base_path = r"{base_path}"
    errors = []
    
    # Test 1: Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"Current directory: {{current_dir}}")
    
    if not current_dir.lower().endswith("projectsplunk"):
        errors.append("Not in projectsplunk directory")
    
    # Test 2: Check r1 directory exists
    r1_path = os.path.join(base_path, "r1")
    print(f"\\nChecking r1 directory: {{r1_path}}")
    if os.path.exists(r1_path):
        print("  [OK] r1 directory exists")
        
        # Check for data files
        data_files = ["device.csv", "http.csv", "logon.csv"]
        for file in data_files:
            file_path = os.path.join(r1_path, file)
            if os.path.exists(file_path):
                print(f"  [OK] {{file}} found")
            else:
                errors.append(f"Missing data file: {{file}}")
    else:
        errors.append("r1 directory not found")
    
    # Test 3: Check checkpoints directory
    checkpoint_dir = os.path.join(r1_path, "checkpoints")
    print(f"\\nChecking checkpoints directory: {{checkpoint_dir}}")
    if os.path.exists(checkpoint_dir):
        print("  [OK] checkpoints directory exists")
    else:
        errors.append("checkpoints directory not found")
    
    # Test 4: Check settings.py has correct path
    settings_file = "insider_threat_detection/config/settings.py"
    if os.path.exists(settings_file):
        print(f"\\nChecking settings.py...")
        with open(settings_file, 'r') as f:
            content = f.read()
        
        if base_path in content:
            print("  [OK] settings.py has correct base path")
        else:
            errors.append("settings.py doesn't have correct path")
    
    # Test 5: Check splunk_credentials.json
    creds_file = "insider_threat_detection/splunk_credentials.json"
    if os.path.exists(creds_file):
        print(f"\\nChecking splunk_credentials.json...")
        with open(creds_file, 'r') as f:
            creds = json.load(f)
        
        if "project_path" in creds:
            if base_path in creds["project_path"]:
                print("  [OK] splunk_credentials.json has correct path")
            else:
                errors.append("splunk_credentials.json has wrong project_path")
    
    # Test 6: Import test
    print("\\nTesting Python imports...")
    try:
        sys.path.insert(0, "insider_threat_detection")
        from config import settings
        print(f"  [OK] Successfully imported settings")
        print(f"  [OK] DATA_PATH = {{settings.DATA_PATH}}")
        
        if settings.DATA_PATH == os.path.join(base_path, "r1"):
            print("  [OK] DATA_PATH is correct")
        else:
            errors.append(f"DATA_PATH is wrong: {{settings.DATA_PATH}}")
    except Exception as e:
        errors.append(f"Import error: {{e}}")
    
    # Summary
    print("\\n" + "=" * 60)
    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  [ERROR] {{error}}")
        print(f"\\nTotal errors: {{len(errors)}}")
        return False
    else:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("All paths are correctly configured.")
        return True

if __name__ == "__main__":
    success = test_paths()
    sys.exit(0 if success else 1)
'''
    
    test_file = "test_all_paths.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    print(f"\nCreated test script: {test_file}")
    print("Run it with: python test_all_paths.py")

if __name__ == "__main__":
    update_all_paths()
