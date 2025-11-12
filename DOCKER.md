# 🐳 CineMatch V2.0 Docker Deployment

Quick and easy Docker deployment for the enhanced multi-algorithm recommendation system.

## 🚀 One-Command Deployment

### Windows (PowerShell)
```powershell
.\deploy.ps1
```

### Linux/macOS (Bash)
```bash
chmod +x deploy.sh
./deploy.sh
```

### Manual Docker Compose
```bash
docker-compose up --build -d
```

## 📱 Access Your Application

Once deployed, access CineMatch V2.0 at: **http://localhost:8501**

## 🔧 Available Algorithms

The V2.0 system includes all 4 recommendation algorithms:

1. **SVD Matrix Factorization** - Fast, accurate for dense users
2. **KNN User-Based** - Social recommendations based on similar users  
3. **KNN Item-Based** - Content-based recommendations
4. **Hybrid (Best of All)** - Intelligent combination of all algorithms ⭐

## 📊 Performance Comparison

| Algorithm | RMSE  | Speed  | Coverage | Memory | Best For |
|-----------|-------|--------|----------|---------|----------|
| SVD       | 0.5690| Fast   | 12.8%    | <1 MB   | Dense users |
| User KNN  | 0.5992| Medium | 100%     | 1 MB    | Sparse users |
| Item KNN  | 1.0218| Slow   | 4.1%     | 51 MB   | Content-based |
| **Hybrid**| **0.5585**| Slower | **100%** | 52 MB   | **All users** ⭐|

## 🎯 V2.0 Features

✨ **Algorithm Selector** - Switch between algorithms in real-time  
📈 **Live Metrics** - See RMSE, training time, memory usage  
🎨 **Professional UI** - Netflix-style interface  
🧠 **Explainable AI** - Understand why movies were recommended  
⚙️ **Advanced Options** - Tune algorithm parameters  

## 🛠️ Management Commands

```bash
# View logs
docker-compose logs -f cinematch-v2

# Stop the application
docker-compose down

# Restart with fresh build
docker-compose down && docker-compose up --build -d

# View container status
docker-compose ps
```

## 📋 Requirements

- Docker Desktop or Docker Engine
- Docker Compose
- 8GB+ RAM recommended for optimal performance
- Port 8501 available

## 🔧 Configuration

The application is configured via environment variables in `docker-compose.yml`:

```yaml
environment:
  - CINEMATCH_VERSION=2.0
  - ENABLE_ALL_ALGORITHMS=true
  - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## 📂 Volume Mounts

- `./data:/app/data` - MovieLens dataset storage
- `./models:/app/models` - Trained model persistence

## 🏥 Health Checks

The container includes health checks that monitor the Streamlit application:
- **Interval**: 30 seconds
- **Timeout**: 10 seconds  
- **Retries**: 3
- **Start Period**: 60 seconds

## 🐛 Troubleshooting

**Container won't start:**
```bash
docker-compose logs cinematch-v2
```

**Port already in use:**
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use port 8502 instead
```

**Memory issues:**
- Reduce sample size in data loading
- Increase Docker memory limits
- Use individual algorithms instead of Hybrid

**Permission issues (Linux/macOS):**
```bash
chmod +x deploy.sh
sudo docker-compose up --build -d
```