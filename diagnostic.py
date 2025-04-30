# Here's a simple diagnostic script you can run to check your environment
# Save this as diagnostic.py in your application directory

import os
import sys
import logging
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'flask', 'langchain_openai', 'langchain_chroma', 
        'langchain_core', 'dotenv', 'openai'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} is NOT installed")
    
    return missing

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = ['OPENAI_API_KEY']
    
    missing = []
    for var in required_vars:
        if os.environ.get(var):
            print(f"✅ {var} is set")
        else:
            missing.append(var)
            print(f"❌ {var} is NOT set")
    
    return missing

def check_file_permissions():
    """Check if all required directories and files have proper permissions."""
    paths = [
        './studio_db',
        './templates',
        './nohup.out'
    ]
    
    issues = []
    for path in paths:
        if not os.path.exists(path):
            issues.append(f"{path} does not exist")
            print(f"❌ {path} does not exist")
            continue
            
        if not os.access(path, os.R_OK):
            issues.append(f"{path} is not readable")
            print(f"❌ {path} is not readable")
        else:
            print(f"✅ {path} is readable")
            
        if os.path.isdir(path) and not os.access(path, os.W_OK):
            issues.append(f"{path} is not writable")
            print(f"❌ {path} is not writable")
        elif os.path.isdir(path):
            print(f"✅ {path} is writable")
    
    return issues

if __name__ == "__main__":
    print("Running diagnostics...")
    print("\nChecking dependencies:")
    missing_deps = check_dependencies()
    
    print("\nChecking environment variables:")
    missing_vars = check_environment()
    
    print("\nChecking file permissions:")
    permission_issues = check_file_permissions()
    
    print("\nDiagnostic Summary:")
    if not missing_deps and not missing_vars and not permission_issues:
        print("✅ All checks passed! Your environment appears to be properly configured.")
    else:
        print("❌ Some issues were found:")
        if missing_deps:
            print(f"  - Missing dependencies: {', '.join(missing_deps)}")
            print("    Fix with: pip install " + " ".join(missing_deps))
        if missing_vars:
            print(f"  - Missing environment variables: {', '.join(missing_vars)}")
            print("    Add these to your .env file")
        if permission_issues:
            print(f"  - Permission issues: {', '.join(permission_issues)}")
            print("    Fix with appropriate chmod commands")