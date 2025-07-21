@echo off
echo ========================================
echo Simple GitHub Upload Fix
echo ========================================

echo.
echo Step 1: Removing projectenv from Git tracking
echo ----------------------------------------
git rm -r --cached projectenv/

echo.
echo Step 2: Setting up Git LFS
echo ----------------------------------------
git lfs install

echo.
echo Step 3: Tracking large files with Git LFS
echo ----------------------------------------
REM Track all CSV files
git lfs track "*.csv"
git lfs track "*.pyd"
git lfs track "*.dll"
git lfs track "*.lib"
git lfs track "*.so"

echo.
echo Step 4: Adding changes
echo ----------------------------------------
git add .gitignore
git add .gitattributes

echo.
echo Step 5: Committing changes
echo ----------------------------------------
git commit -m "Fix: Remove virtual environment and setup Git LFS for large files"

echo.
echo Step 6: Resetting to remove large files from history
echo ----------------------------------------
echo WARNING: This will reset your repository to remove large files from history
echo Press Ctrl+C to cancel, or
pause

REM Create a new branch without the large files
git checkout --orphan temp_branch
git add -A
git commit -m "Initial commit with Git LFS and without virtual environment"
git branch -D main
git branch -m main

echo.
echo ========================================
echo DONE! Now push to GitHub:
echo ========================================
echo.
echo Run: git push --force origin main
echo.
echo If you get LFS errors, first run:
echo git lfs push --all origin
echo.
pause
