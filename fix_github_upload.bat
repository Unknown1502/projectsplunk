@echo off
echo ========================================
echo Fixing GitHub Upload Issues
echo ========================================

echo.
echo Step 1: Installing Git LFS (if not already installed)
echo ----------------------------------------
git lfs install

echo.
echo Step 2: Creating .gitattributes for Git LFS
echo ----------------------------------------
echo # Git LFS tracking for large files > .gitattributes
echo *.csv filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.pyd filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.dll filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.lib filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.so filter=lfs diff=lfs merge=lfs -text >> .gitattributes

echo.
echo Step 3: Removing projectenv from Git tracking
echo ----------------------------------------
git rm -r --cached projectenv/

echo.
echo Step 4: Removing large files from Git history
echo ----------------------------------------
echo This will use git filter-branch to remove large files from history

REM Remove the virtual environment completely from history
git filter-branch --force --index-filter "git rm -r --cached --ignore-unmatch projectenv/" --prune-empty --tag-name-filter cat -- --all

echo.
echo Step 5: Adding large CSV files to Git LFS
echo ----------------------------------------
git lfs track "insider_threat_detection_app/lookups/training_http.csv"
git lfs track "insider_threat_detection_app/lookups/training_logon.csv"
git lfs track "r1/*.csv"

echo.
echo Step 6: Committing changes
echo ----------------------------------------
git add .gitignore
git add .gitattributes
git add insider_threat_detection_app/lookups/training_http.csv
git add insider_threat_detection_app/lookups/training_logon.csv
git commit -m "Fix: Add .gitignore, setup Git LFS for large files, remove virtual environment from tracking"

echo.
echo Step 7: Cleaning up Git repository
echo ----------------------------------------
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo.
echo ========================================
echo IMPORTANT: Next Steps
echo ========================================
echo.
echo 1. The repository has been cleaned. You need to force push:
echo    git push --force origin main
echo.
echo 2. If you get an error about Git LFS, you may need to:
echo    - Install Git LFS: https://git-lfs.github.com/
echo    - Run: git lfs install
echo    - Then run this script again
echo.
echo 3. Your virtual environment (projectenv) is now ignored by Git
echo    but still exists locally.
echo.
pause
