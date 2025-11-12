#!/bin/bash

# CineMatch V2.0 Multi-Algorithm Deployment Script
# One-command deployment for the enhanced recommendation system

echo "🎬 CineMatch V2.0 Multi-Algorithm Deployment"
echo "============================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker Compose is available"

# Stop any existing containers
echo "🔄 Stopping existing CineMatch containers..."
docker-compose down 2>/dev/null || docker compose down 2>/dev/null || true

# Build and start the V2.0 system
echo "🚀 Building and starting CineMatch V2.0 Multi-Algorithm System..."

if command -v docker-compose &> /dev/null; then
    docker-compose up --build -d
else
    docker compose up --build -d
fi

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! CineMatch V2.0 is now running!"
    echo ""
    echo "📱 Access your multi-algorithm recommendation system:"
    echo "   🌐 Web Interface: http://localhost:8501"
    echo ""
    echo "🔧 Available Algorithms:"
    echo "   📊 SVD Matrix Factorization"
    echo "   👥 KNN User-Based Collaborative Filtering" 
    echo "   🎬 KNN Item-Based Content Filtering"
    echo "   🚀 Hybrid (Best of All)"
    echo ""
    echo "📊 To view logs: docker-compose logs -f cinematch-v2"
    echo "🛑 To stop: docker-compose down"
    echo ""
    echo "🎯 Features in V2.0:"
    echo "   ✨ Algorithm selector with live switching"
    echo "   📈 Real-time performance metrics (RMSE, speed, memory)"
    echo "   🎨 Professional Netflix-style interface"
    echo "   🧠 Explainable AI - see why movies were recommended"
    echo "   ⚙️ Advanced parameter tuning options"
    echo ""
else
    echo "❌ Deployment failed. Check the logs with:"
    echo "   docker-compose logs cinematch-v2"
fi