# CineMatch V2.0 Multi-Algorithm Deployment Script
# One-command deployment for the enhanced recommendation system

Write-Host "🎬 CineMatch V2.0 Multi-Algorithm Deployment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker info *>$null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if Docker Compose is available
$composeAvailable = $false
try {
    docker-compose version *>$null
    $composeAvailable = $true
    $composeCommand = "docker-compose"
} catch {
    try {
        docker compose version *>$null
        $composeAvailable = $true
        $composeCommand = "docker compose"
    } catch {
        Write-Host "❌ Docker Compose is not available. Please install Docker Compose." -ForegroundColor Red
        exit 1
    }
}

if ($composeAvailable) {
    Write-Host "✅ Docker Compose is available" -ForegroundColor Green
}

# Stop any existing containers
Write-Host "🔄 Stopping existing CineMatch containers..." -ForegroundColor Yellow
try {
    & $composeCommand.Split() down *>$null
} catch {
    # Ignore errors if no containers are running
}

# Build and start the V2.0 system
Write-Host "🚀 Building and starting CineMatch V2.0 Multi-Algorithm System..." -ForegroundColor Yellow

$buildArgs = $composeCommand.Split() + @("up", "--build", "-d")
& $buildArgs[0] $buildArgs[1..($buildArgs.Length-1)]

# Check if deployment was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 SUCCESS! CineMatch V2.0 is now running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📱 Access your multi-algorithm recommendation system:" -ForegroundColor Cyan
    Write-Host "   🌐 Web Interface: " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:8501" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔧 Available Algorithms:" -ForegroundColor Cyan
    Write-Host "   📊 SVD Matrix Factorization" -ForegroundColor White
    Write-Host "   👥 KNN User-Based Collaborative Filtering" -ForegroundColor White
    Write-Host "   🎬 KNN Item-Based Content Filtering" -ForegroundColor White
    Write-Host "   🚀 Hybrid (Best of All)" -ForegroundColor White
    Write-Host ""
    Write-Host "📊 To view logs: " -NoNewline -ForegroundColor Cyan
    Write-Host "$composeCommand logs -f cinematch-v2" -ForegroundColor Yellow
    Write-Host "🛑 To stop: " -NoNewline -ForegroundColor Cyan  
    Write-Host "$composeCommand down" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🎯 Features in V2.0:" -ForegroundColor Cyan
    Write-Host "   ✨ Algorithm selector with live switching" -ForegroundColor White
    Write-Host "   📈 Real-time performance metrics (RMSE, speed, memory)" -ForegroundColor White
    Write-Host "   🎨 Professional Netflix-style interface" -ForegroundColor White
    Write-Host "   🧠 Explainable AI - see why movies were recommended" -ForegroundColor White
    Write-Host "   ⚙️ Advanced parameter tuning options" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "❌ Deployment failed. Check the logs with:" -ForegroundColor Red
    Write-Host "   $composeCommand logs cinematch-v2" -ForegroundColor Yellow
}