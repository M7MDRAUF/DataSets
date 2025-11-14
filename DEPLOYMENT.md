# 🚀 CineMatch V2.1.0 - Production Deployment Guide

## ⚠️ Important: Recommended Deployment Platform

**Use Streamlit Cloud** (not Vercel) - Streamlit apps work best on Streamlit's native platform.

**V2.1.0 Critical**: Pre-trained models (526MB total: User-KNN 266MB, Item-KNN 260MB, Content-Based ~300MB) require **Git LFS**. All deployment methods MUST support Git LFS for model files.

Vercel has limitations with:
- Long-running Python processes
- Large dependencies (scikit-surprise, numba, scipy)
- Streamlit's websocket connections
- Git LFS (large model files)

---

## 🔑 Prerequisites: Git LFS Setup (REQUIRED)

**ALL deployment methods require Git LFS for pre-trained models (526MB)**

### Install Git LFS

**Windows:**
```powershell
# Download from https://git-lfs.github.com/ or use Chocolatey
choco install git-lfs
git lfs install
```

**macOS:**
```bash
brew install git-lfs
git lfs install
```

**Linux:**
```bash
sudo apt-get install git-lfs  # Ubuntu/Debian
sudo yum install git-lfs      # CentOS/RHEL
git lfs install
```

### Pull Pre-trained Models

```bash
# After cloning repository
cd DataSets
git lfs pull

# Verify models downloaded (should see 526MB total)
ls -lh models/
# Expected files:
# - user_knn_model.pkl (266 MB)
# - item_knn_model.pkl (260 MB)
# - content_based_*.pkl (~300 MB)
```

**Without Git LFS**: Models will be pointer files (~100 bytes) and application will fail with "model not found" errors.

---

## Deployment Options

### Option 1: Deploy to Streamlit Cloud (✅ RECOMMENDED)

**Current Live Deployment**: https://m7md007.streamlit.app

#### Prerequisites
- GitHub account
- Streamlit Cloud account (free tier available)
- Git LFS enabled on repository (already configured)

#### Steps:

1. **Your repository is already on GitHub**
   - Repository: https://github.com/M7MDRAUF/DataSets
   - Branch: `main`
   - Git LFS: ✅ Configured (526MB models)

2. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `M7MDRAUF/DataSets`
   - Set main file: `app/main.py`
   - Branch: `main`
   - Click "Deploy"
   - **Important**: Streamlit Cloud automatically pulls Git LFS files

3. **Verify Deployment**
   - Check logs for "AlgorithmManager initialized" message
   - Confirm all 5 algorithms load successfully:
     - ✅ SVD (trained on startup)
     - ✅ User-KNN (pre-trained, 266MB)
     - ✅ Item-KNN (pre-trained, 260MB)
     - ✅ Content-Based (pre-trained, ~300MB)
     - ✅ Hybrid (ensemble of all 4)

4. **Access your app**
   - URL: `https://YOUR-APP-NAME.streamlit.app`
   - Example: https://m7md007.streamlit.app

---

### Option 2: Docker Deployment (Self-Hosted)

**Best for**: Self-hosted deployments, cloud VMs (AWS, GCP, Azure), on-premises servers

#### Prerequisites
- Docker and Docker Compose installed
- Git LFS installed (see prerequisites above)
- Server with minimum 4GB RAM (8GB recommended)
- 3GB free disk space (600MB dataset + 526MB models + containers)

#### Steps:

1. **Clone the repository with LFS**
   ```bash
   git clone https://github.com/M7MDRAUF/DataSets.git
   cd DataSets
   
   # Pull pre-trained models (CRITICAL)
   git lfs pull
   
   # Verify models downloaded (should show ~526MB)
   ls -lh models/
   ```

2. **Download the MovieLens 32M dataset**
   ```bash
   # Download and extract
   wget http://files.grouplens.org/datasets/movielens/ml-32m.zip
   unzip ml-32m.zip
   
   # Move files to correct location
   mkdir -p data/ml-32m
   mv ml-32m/*.csv data/ml-32m/
   
   # Verify dataset (should have 4 CSV files)
   ls data/ml-32m/
   # Expected: ratings.csv, movies.csv, links.csv, tags.csv
   ```

3. **Build and run Docker container**
   ```bash
   docker-compose up --build
   
   # Or run in background
   docker-compose up --build -d
   ```

4. **Access the application**
   - Open browser: **http://localhost:8501**
   - Wait for "AlgorithmManager initialized" in logs
   - All 5 algorithms should load successfully

---

## 📦 Dataset Setup

**Important**: Due to file size limits, datasets are NOT included in the repository.

### Download Dataset:

```bash
# Option 1: Manual download
# Visit: https://grouplens.org/datasets/movielens/32m/
# Download ml-32m.zip
# Extract to: data/ml-32m/

# Option 2: Command line (Linux/Mac)
cd data
wget http://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip
mv ml-32m/* ml-32m/
rm -rf ml-32m.zip

# Option 3: PowerShell (Windows)
cd data
Invoke-WebRequest -Uri "http://files.grouplens.org/datasets/movielens/ml-32m.zip" -OutFile "ml-32m.zip"
Expand-Archive -Path "ml-32m.zip" -DestinationPath "."
Move-Item -Path "ml-32m\*" -Destination "ml-32m\" -Force
Remove-Item "ml-32m.zip"
```

### Required Files:
- `data/ml-32m/ratings.csv` (877 MB)
- `data/ml-32m/movies.csv` (2.5 MB)
- `data/ml-32m/tags.csv` (34 MB)
- `data/ml-32m/links.csv` (3 MB)

---

## 🔧 Environment Configuration

### Required Environment Variables

Create a `.env` file (optional, for custom configuration):

```env
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Model Configuration
MODEL_PATH=models/svd_model_optimized.pkl
MODEL_RETRAIN=false

# Data Configuration
DATA_PATH=data/ml-32m/
MAX_RATINGS_LOAD=32000000
```

---

## 🌐 Domain Configuration (Optional)

### Custom Domain on Vercel

1. Go to your Vercel project → Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed
4. Wait for SSL certificate provisioning

### Custom Domain on Streamlit Cloud

1. Contact Streamlit support for custom domain setup
2. Or use Streamlit's provided subdomain: `your-app.streamlit.app`

---

## 📊 Production Checklist

Before deploying to production:

- [ ] Dataset downloaded and placed in correct directory
- [ ] Model trained and saved (or download pre-trained model)
- [ ] Dependencies installed from `requirements.txt`
- [ ] Environment variables configured
- [ ] .gitignore properly excludes large files
- [ ] README updated with deployment instructions
- [ ] Tests passing locally
- [ ] Application runs without errors locally
- [ ] Docker build succeeds (if using Docker)
- [ ] Repository pushed to GitHub
- [ ] Deployment platform configured
- [ ] SSL/TLS certificate active (for custom domains)

---

## 🔍 Monitoring & Maintenance

### Health Checks

```bash
# Check if app is running
curl http://localhost:8501/_stcore/health

# Check Docker container status
docker-compose ps

# View logs
docker-compose logs -f
```

### Common Issues

**Issue**: App crashes on startup
- **Solution**: Check logs, ensure dataset is downloaded, verify model file exists

**Issue**: Slow response times
- **Solution**: Increase server resources, enable caching, use CDN for static assets

**Issue**: Out of memory errors
- **Solution**: Increase container/server memory, optimize data loading

---

## 📞 Support

For deployment issues:
- Check [GitHub Issues](https://github.com/YOUR_USERNAME/cinematch/issues)
- Review [Documentation](DOCUMENTATION_INDEX.md)
- Contact: [Your contact information]

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

**Last Updated**: October 27, 2025
