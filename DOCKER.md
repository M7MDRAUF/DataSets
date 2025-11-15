# 🐳 CineMatch V2.1.1 Docker Deployment

Quick and easy Docker deployment for the complete 5-algorithm recommendation system with optimized memory usage.

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

Once deployed, access CineMatch V2.1.1 at: **http://localhost:8501**

## 🔧 Available Algorithms

The V2.1.1 system includes all 5 recommendation algorithms:

1. **SVD Matrix Factorization (sklearn)** - Fast, accurate for dense users (909.6 MB, RMSE 0.7502)
2. **User-KNN (User-Based CF)** - Social recommendations based on similar users (1114 MB, RMSE 0.8394)
3. **Item-KNN (Item-Based CF)** - Item similarity recommendations (1108.4 MB, RMSE 0.9100)
4. **Content-Based Filtering** - TF-IDF feature matching (genres, tags, titles) (1059.9 MB, RMSE 1.1130)
5. **Hybrid Ensemble** - Intelligent combination of all 4 algorithms (491.3 MB, RMSE 0.8701) ⭐

## 📊 Performance Comparison (V2.1.1)

| Algorithm | Model Size | Load Time | RMSE | Coverage | Best For |
|-----------|-----------|-----------|------|----------|----------|
| SVD (sklearn) | 909.6 MB | 5-9s | 0.7502 | 100% | Dense users |
| User-KNN | 1114 MB | 1-8s | 0.8394 | 100% | Sparse users |
| Item-KNN | 1108.4 MB | 1-8s | 0.9100 | 50.1% | Item similarity |
| Content-Based | 1059.9 MB | ~6s | 1.1130 | 100% | New items/cold-start |
| **Hybrid** | **491.3 MB** | **~25s** | **0.8701** | **100%** | **All scenarios** ⭐|

**Total Models**: 4.07 GB on disk

## 🎯 V2.1.1 Features

✨ **5 Algorithms** - SVD, User-KNN, Item-KNN, Content-Based, Hybrid  
🔄 **Algorithm Selector** - Switch between algorithms in real-time  
📈 **Live Metrics** - See RMSE, training time, memory usage  
🎨 **Professional UI** - Netflix-style interface (clean, no debug spam)  
🧠 **Explainable AI** - Understand why movies were recommended  
📊 **Analytics Dashboard** - 5 tabs with comprehensive benchmarking  
🚀 **Memory Optimized** - 98.6% reduction (13.2GB → 185MB runtime)

## 💾 Memory Architecture (V2.1.1)

**Runtime Memory Usage**:
- Application: **185 MB** (shallow references optimization)
- Docker Container: **2.6 GB total** (32% of 8GB limit)
- Headroom: **5.4 GB free** (68% available)

**Key Optimization**:
- Shallow references for pre-trained models (read-only)
- One-time data initialization in Streamlit session
- Thread-safe algorithm caching
- Smart garbage collection

**Docker Memory Limit**: 8GB (configured in docker-compose.yml)

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

# Check memory usage
docker stats cinematch-v2
```

## 🧹 Complete Docker Cleanup (V2.1.1)

When you need to start fresh or reclaim disk space:

```bash
# Stop and remove containers, volumes, networks
docker-compose down -v

# Remove all Docker resources (images, cache, build cache)
docker system prune -af --volumes

# Verify cleanup
docker system df

# Fresh rebuild
docker-compose up --build -d
```

**Expected Cleanup**: ~2-4 GB reclaimed

## 📋 Requirements

- Docker Desktop or Docker Engine
- Docker Compose
- **8GB RAM minimum** (for 4.07GB models + runtime)
- 8GB+ RAM recommended for optimal performance
- Port 8501 available

## 🔧 Configuration

The application is configured via environment variables in `docker-compose.yml`:

```yaml
environment:
  - CINEMATCH_VERSION=2.1.0
  - ENABLE_ALL_ALGORITHMS=true
  - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
  - STREAMLIT_SERVER_PORT=8501
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