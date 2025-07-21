# GitHub Upload Fix Guide

## Problem Summary
Your repository contains files that exceed GitHub's file size limits:
- Maximum file size: 100 MB
- Files exceeding this limit cannot be pushed to GitHub
- Your virtual environment (`projectenv/`) contains large Python packages

## Files Causing Issues

### Files > 100 MB (Must be removed or use Git LFS):
1. `projectenv/Lib/site-packages/jaxlib/xla_extension.pyd` (119.94 MB)
2. `projectenv/Lib/site-packages/tensorflow/python/_pywrap_tensorflow_internal.pyd` (641.06 MB)
3. `projectenv/Lib/site-packages/~ensorflow/python/_pywrap_tensorflow_internal.pyd` (697.64 MB)
4. `insider_threat_detection_app/lookups/training_http.csv` (290.15 MB)

### Files > 50 MB (Warning, but can be pushed):
1. `projectenv/Lib/site-packages/llvmlite/binding/llvmlite.dll` (84.52 MB)
2. `projectenv/Lib/site-packages/tensorflow/python/_pywrap_tensorflow_internal.lib` (50.72 MB)
3. `insider_threat_detection_app/lookups/training_logon.csv` (58.70 MB)
4. `projectenv/Lib/site-packages/clang/native/libclang.dll` (80.10 MB)

## Solution Options

### Option 1: Quick Fix (Recommended)
Run the `simple_github_fix.bat` script which will:
1. Remove the virtual environment from Git tracking
2. Set up Git LFS for large files
3. Create a clean commit history

```bash
# Run this in your project directory
simple_github_fix.bat
```

### Option 2: Manual Steps

#### Step 1: Install Git LFS
```bash
# Download from https://git-lfs.github.com/
# Or use package manager:
# Windows: choco install git-lfs
# After installation:
git lfs install
```

#### Step 2: Remove Virtual Environment from Git
```bash
# Remove from tracking (keeps local files)
git rm -r --cached projectenv/

# Add to .gitignore (already done)
```

#### Step 3: Set up Git LFS for Large Files
```bash
# Track large file types
git lfs track "*.csv"
git lfs track "*.pyd"
git lfs track "*.dll"
git lfs track "*.lib"
git lfs track "*.so"

# This creates/updates .gitattributes
```

#### Step 4: Clean Git History (Choose one method)

**Method A: Create New Clean History (Simplest)**
```bash
# Create orphan branch
git checkout --orphan temp_branch

# Add all files
git add -A

# Commit
git commit -m "Initial commit with Git LFS"

# Delete old main branch
git branch -D main

# Rename temp branch to main
git branch -m main

# Force push
git push --force origin main
```

**Method B: Filter Branch (Preserves some history)**
```bash
# Remove projectenv from all history
git filter-branch --force --index-filter \
  "git rm -r --cached --ignore-unmatch projectenv/" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push
git push --force origin main
```

### Option 3: Alternative Solutions

#### Use GitHub Releases for Large Files
Instead of tracking large CSV files in the repository:
1. Upload them as GitHub Release assets
2. Document how to download them
3. Add download scripts to your repository

#### Split Large Files
For CSV files, you could:
1. Split them into smaller chunks
2. Compress them (if not already compressed)
3. Store only samples in the repository

## After Fixing

### Verify Success
```bash
# Check file sizes in repository
git ls-files -s | awk '{print $4}' | xargs -I {} sh -c 'echo -n "{}: "; du -h "{}"' | sort -h

# Push to GitHub
git push --force origin main

# If using Git LFS, also run:
git lfs push --all origin
```

### Best Practices Going Forward

1. **Never commit virtual environments**
   - Always add `venv/`, `env/`, `projectenv/` to `.gitignore`
   - Use `requirements.txt` or `Pipfile` instead

2. **Handle large data files carefully**
   - Use Git LFS for files > 50 MB
   - Consider cloud storage for very large datasets
   - Keep only sample data in the repository

3. **Check before committing**
   ```bash
   # Check for large files before committing
   find . -type f -size +50M | grep -v ".git"
   ```

## Troubleshooting

### Error: "git lfs" is not recognized
- Install Git LFS from https://git-lfs.github.com/

### Error: "failed to push some refs"
- You need to force push: `git push --force origin main`
- Make sure you have the correct permissions

### Error: "This repository is over its data quota"
- You may need to purchase additional Git LFS storage
- Or remove some large files from LFS

## Your GitHub Credentials
- GitHub ID: Unknown1502
- Repository: https://github.com/Unknown1502/projectsplunk.git

**Note**: Never share your password publicly. Consider using:
- GitHub Personal Access Tokens
- SSH keys for authentication
- GitHub CLI for easier authentication
