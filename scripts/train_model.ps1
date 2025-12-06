# CineMatch V2.1.6 - Model Training Script (PowerShell)
# Wrapper script for training the SVD model on Windows

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "CineMatch V2.1.6 - Model Training" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Check if virtual environment is activated (recommended)
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Warning: No virtual environment detected" -ForegroundColor Yellow
    Write-Host "   It's recommended to use a virtual environment" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        exit 1
    }
}

# Run the training script
Write-Host ""
Write-Host "🚀 Starting model training..." -ForegroundColor Cyan
Write-Host ""

python src/model_training.py

# Check exit status
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "✅ Training completed successfully!" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Run the app: streamlit run app/main.py"
    Write-Host "2. Or use Docker: docker-compose up"
} else {
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Red
    Write-Host "❌ Training failed!" -ForegroundColor Red
    Write-Host "=========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above" -ForegroundColor Yellow
    exit 1
}
