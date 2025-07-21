# PowerShell script to fix GitHub upload issues
Write-Host "========================================"
Write-Host "Fixing GitHub Upload Issues"
Write-Host "========================================"

Write-Host "`nStep 1: Checking Git LFS installation"
Write-Host "----------------------------------------"
$lfsInstalled = git lfs version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Git LFS is not installed. Please install it from: https://git-lfs.github.com/" -ForegroundColor Red
    Write-Host "After installation, run this script again."
    exit 1
} else {
    Write-Host "Git LFS is installed: $lfsInstalled" -ForegroundColor Green
    git lfs install
}

Write-Host "`nStep 2: Creating .gitattributes for Git LFS"
Write-Host "----------------------------------------"
@"
# Git LFS tracking for large files
*.csv filter=lfs diff=lfs merge=lfs -text
*.pyd filter=lfs diff=lfs merge=lfs -text
*.dll filter=lfs diff=lfs merge=lfs -text
*.lib filter=lfs diff=lfs merge=lfs -text
*.so filter=lfs diff=lfs merge=lfs -text
"@ | Out-File -FilePath ".gitattributes" -Encoding UTF8

Write-Host "`nStep 3: Removing projectenv from Git tracking"
Write-Host "----------------------------------------"
git rm -r --cached projectenv/ 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Removed projectenv from Git tracking" -ForegroundColor Green
} else {
    Write-Host "projectenv was not tracked or already removed" -ForegroundColor Yellow
}

Write-Host "`nStep 4: Using BFG Repo-Cleaner to remove large files from history"
Write-Host "----------------------------------------"
Write-Host "Downloading BFG Repo-Cleaner..."

# Download BFG if not exists
if (-not (Test-Path "bfg.jar")) {
    Invoke-WebRequest -Uri "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar" -OutFile "bfg.jar"
    Write-Host "Downloaded BFG Repo-Cleaner" -ForegroundColor Green
}

# Create a backup
Write-Host "`nCreating backup of your repository..."
$backupName = "projectsplunk_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Copy-Item -Path ".git" -Destination "../$backupName" -Recurse
Write-Host "Backup created at: ../$backupName" -ForegroundColor Green

Write-Host "`nRemoving files larger than 50M from history..."
java -jar bfg.jar --strip-blobs-bigger-than 50M

Write-Host "`nStep 5: Cleaning up repository"
Write-Host "----------------------------------------"
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Host "`nStep 6: Tracking large CSV files with Git LFS"
Write-Host "----------------------------------------"
git lfs track "insider_threat_detection_app/lookups/*.csv"
git lfs track "r1/*.csv"

Write-Host "`nStep 7: Adding and committing changes"
Write-Host "----------------------------------------"
git add .gitignore
git add .gitattributes

# Check if there are large CSV files to add
$csvFiles = @(
    "insider_threat_detection_app/lookups/training_http.csv",
    "insider_threat_detection_app/lookups/training_logon.csv"
)

foreach ($file in $csvFiles) {
    if (Test-Path $file) {
        git add $file
        Write-Host "Added $file to Git LFS" -ForegroundColor Green
    }
}

git commit -m "Fix: Setup Git LFS, add .gitignore, remove virtual environment from tracking"

Write-Host "`n========================================"
Write-Host "COMPLETED! Next Steps:" -ForegroundColor Green
Write-Host "========================================"
Write-Host ""
Write-Host "1. Force push to GitHub:" -ForegroundColor Yellow
Write-Host "   git push --force origin main"
Write-Host ""
Write-Host "2. If you get LFS errors, run:" -ForegroundColor Yellow
Write-Host "   git lfs push --all origin"
Write-Host ""
Write-Host "3. Your virtual environment (projectenv) is now ignored" -ForegroundColor Cyan
Write-Host "   but still exists locally."
Write-Host ""
Write-Host "4. A backup of your .git folder was created at:" -ForegroundColor Cyan
Write-Host "   ../$backupName"
Write-Host ""

Read-Host "Press Enter to continue..."
