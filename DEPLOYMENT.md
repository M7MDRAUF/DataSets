# 🚀 CineMatch - Production Deployment Guide

## ⚠️ Important: Recommended Deployment Platform

**Use Streamlit Cloud** (not Vercel) - Streamlit apps work best on Streamlit's native platform.

Vercel has limitations with:
- Long-running Python processes
- Large dependencies (scikit-surprise, numba)
- Streamlit's websocket connections

---

## Deployment Options

### Option 1: Deploy to Streamlit Cloud (✅ RECOMMENDED)

#### Prerequisites
- GitHub account
- Streamlit Cloud account (free tier available)

#### Steps:

1. **Your repository is already on GitHub**
   - Repository: https://github.com/M7MDRAUF/DataSets
   - Branch: `main`

2. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `M7MDRAUF/DataSets`
   - Set main file: `app/main.py`
   - Branch: `main`
   - Click "Deploy"

3. **Configure Secrets** (if needed)
   - Go to app settings → Secrets
   - Add any required environment variables

4. **Access your app**
   - URL will be: `https://YOUR-APP-NAME.streamlit.app`

---

### Option 2: Deploy to Heroku (Alternative)

#### Prerequisites
- GitHub account
- Streamlit Cloud account (free tier available)

#### Steps:

1. **Push to GitHub** (same as above)

2. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Click "New app"
   - Select your repository
   - Set main file: `app/main.py`
   - Click "Deploy"

3. **Configure Secrets** (if needed)
   - Go to app settings → Secrets
   - Add any required environment variables

---

### Option 3: Docker Deployment (Most Flexible)

#### Prerequisites
- Docker and Docker Compose installed
- Server with minimum 4GB RAM

#### Steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/cinematch.git
   cd cinematch
   ```

2. **Download the dataset**
   ```bash
   # Download MovieLens 32M dataset
   wget http://files.grouplens.org/datasets/movielens/ml-32m.zip
   unzip ml-32m.zip -d data/
   mv data/ml-32m/* data/ml-32m/
   ```

3. **Build and run**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   - Open browser: http://localhost:8501

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
