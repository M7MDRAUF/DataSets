# 🐳 CineMatch V2.1.0 Docker Deployment

Quick and easy Docker deployment for the complete 5-algorithm recommendation system.

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

Once deployed, access CineMatch V2.1.0 at: **http://localhost:8501**

## 🔧 Available Algorithms

The V2.1.0 system includes all 5 recommendation algorithms:

1. **SVD Matrix Factorization** - Fast, accurate for dense users
2. **User-KNN (User-Based CF)** - Social recommendations based on similar users  
3. **Item-KNN (Item-Based CF)** - Item similarity recommendations
4. **Content-Based Filtering** - TF-IDF feature matching (genres, tags, titles)
5. **Hybrid Ensemble** - Intelligent combination of all 4 algorithms ⭐

## 📊 Performance Comparison

| Algorithm | RMSE (100K) | Speed  | Coverage | Memory | Best For |
|-----------|-------------|--------|----------|---------|----------|
| SVD       | 0.6829      | Fast   | 24.1%    | ~50 MB  | Dense users |
| User-KNN  | 0.8394      | Medium | ~50%     | 266 MB  | Sparse users |
| Item-KNN  | 0.8117      | Medium | ~50%     | 260 MB  | Item similarity |
| Content-Based | N/A*    | Fast   | 100%     | ~300 MB | New items/cold-start |
| **Hybrid**| **0.7668**  | Slower | **100%** | 526 MB+ | **All scenarios** ⭐|

*Content-Based uses cosine similarity (not RMSE-based)

**Note**: SVD RMSE on full 32M dataset: 0.8406 (per model_metadata.json)

## 🎯 V2.1.0 Features

✨ **5 Algorithms** - SVD, User-KNN, Item-KNN, Content-Based, Hybrid  
🔄 **Algorithm Selector** - Switch between algorithms in real-time  
📈 **Live Metrics** - See RMSE, training time, memory usage  
🎨 **Professional UI** - Netflix-style interface  
🧠 **Explainable AI** - Understand why movies were recommended  
⚙️ **Advanced Options** - Tune dataset size and parameters  
📊 **Analytics Dashboard** - 5 tabs with comprehensive benchmarking  

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