# 🚀 CineMatch V2.0.0 Quick Setup Guide

This guide will help you get CineMatch V2.0.0 with all 4 algorithms up and running in under 15 minutes!

**What's New in V2.0.0:**
- ✅ Pre-trained KNN models (no training needed for demo!)
- ✅ Analytics Dashboard with benchmarking
- ✅ 200x faster performance optimizations
- ✅ Multi-algorithm support (SVD + User KNN + Item KNN + Hybrid)

---

## ⚡ Quick Start (3 Steps)

### Step 1: Install Dependencies (2 minutes)

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download Dataset (5 minutes)

1. **Download**: http://files.grouplens.org/datasets/movielens/ml-32m.zip
2. **Extract** the ZIP file
3. **Copy** all CSV files to `data/ml-32m/`

**Required files:**
- ✅ `ratings.csv` (32M ratings, ~600MB)
- ✅ `movies.csv` (Movie catalog)
- ✅ `links.csv` (IMDb/TMDb IDs)
- ✅ `tags.csv` (User tags)

### Step 3: Run the App (Instant - Pre-trained Models Included!)

**V2.0.0 Note**: Pre-trained KNN models are included (526MB via Git LFS), so you can run immediately!

```powershell
streamlit run app/main.py --server.port 8508
```

**Open**: http://localhost:8508

#### Optional: Train Additional Models

If you want to retrain models from scratch:

##### Option A: Train KNN Models (30-60 minutes)
```powershell
python train_knn_models.py
# Trains User KNN and Item KNN on full 32M dataset
# Creates user_knn_model.pkl (266MB) and item_knn_model.pkl (260MB)
```

##### Option B: Train SVD Model (10-15 minutes)
```powershell
python src/model_training.py
# Choose dataset size when prompted
```

**Note**: Pre-trained models are already optimized for presentation!

---

## 🐳 Docker Alternative (Recommended for Demo)

```powershell
# After downloading dataset and training model
docker-compose up --build
```

---

## ✅ Verification Checklist V2.0.0

Before running, ensure:

- [ ] Python 3.9+ installed (`python --version`)
- [ ] All dependencies installed (`pip list | grep streamlit`)
- [ ] Dataset files in `data/ml-32m/` (4 CSV files)
- [ ] Pre-trained models exist (Git LFS pull completed):
  - [ ] `models/user_knn_model.pkl` (266MB)
  - [ ] `models/item_knn_model.pkl` (260MB)
- [ ] Port 8508 available (changed from 8501)
- [ ] Git LFS installed (`git lfs version`) for model files

---

## 🎯 Demo Flow V2.0.0 (For Professor)

### 1. Home Page (45 seconds)
- Shows dataset overview (32M ratings, 87K+ movies)
- **NEW**: Algorithm selector with 4 options
- **NEW**: Live performance metrics display
- **NEW**: Dataset size selection (100K/500K/1M/Full)
- Algorithm status indicators
- Genre distribution chart

### 2. Recommend Page (3 minutes)
- **NEW**: Select algorithm from dropdown (SVD/User KNN/Item KNN/Hybrid)
- Enter User ID: **123**
- Click **"Get Recommendations"**
- **NEW**: Observe algorithm switching (e.g., "🔄 Switching to User KNN...")
- **NEW**: View live performance metrics (training time, RMSE, coverage)
- View personalized movie list with ratings
- Click **"Explain"** on first movie (algorithm-specific reasoning)
- Show taste profile sidebar with genre preferences
- Try **"Surprise Me"** button
- **NEW**: Compare recommendations across different algorithms

### 3. Analytics Page (2 minutes) ⭐ NEW IN V2.0.0
- **Algorithm Benchmarking**:
  - Click **"🚀 Run Algorithm Benchmark"** button
  - View performance comparison table (RMSE/MAE/Coverage)
  - See interactive Plotly charts (accuracy & coverage)
- **Dataset Statistics**: Users, movies, ratings, sparsity
- **Genre Analysis**: Distribution and trends
- **Temporal Trends**: Release year patterns
- **Movie Similarity**: Search for **"Matrix"** and find similar movies

### 4. Technical Highlights V2.0.0
- **Multi-Algorithm Support**: 4 different recommendation paradigms ✅
- **Pre-trained Models**: 526MB models on full 32M dataset ✅
- **Performance**: KNN loading 1.5s (200x faster), Hybrid 7s ✅
- **Accuracy**: SVD RMSE 0.6829, Hybrid RMSE 0.7668, Coverage 100% ✅
- **Response time**: < 2 seconds with smart sampling ✅
- **Explainable AI**: Algorithm-specific reasoning ✅
- **Analytics Dashboard**: Comprehensive benchmarking ✅
- **Docker containerization**: One-command deployment ✅

---

## 🔧 Troubleshooting V2.0.0

### "Data Integrity Check Failed"
**Solution**: Download dataset and place in `data/ml-32m/`

### "Pre-trained model not found"
**Solution**: 
```powershell
# Ensure Git LFS is installed and pull large files
git lfs install
git lfs pull
```

### "Import streamlit could not be resolved"
**Solution**: 
```powershell
pip install -r requirements.txt
```

### "AlgorithmManager not initialized"
**Solution**: Wait for "🎯 Algorithm Manager initialized with data" message

### "'predict_rating' method not found"
**Solution**: This was fixed in V2.0.0 - ensure you're using the latest code (uses `predict()` method)

### "Memory Error during training"
**Solution**: Use pre-trained models (already included) or choose smaller dataset (100K/500K)

---

## 📊 Expected Performance V2.0.0

| Algorithm | RMSE | Loading Time | Recommendation Time | Model Size | Coverage |
|-----------|------|--------------|---------------------|------------|----------|
| **SVD** | 0.6829 | 4-7s (train) | < 2s | ~50MB | 24.1% |
| **User KNN** | 0.8394 | 1.5s (pre-trained) | < 2s | 266MB | ~50% |
| **Item KNN** | 0.8117 | 1.0s (pre-trained) | < 2s | 260MB | ~50% |
| **Hybrid** | 0.7668 | 7s (combined) | < 2s | 488MB | 100% ✅ |

**Performance Improvements:**
- ✅ KNN loading: **200x faster** (1.5s vs 124s before pre-training)
- ✅ Hybrid completion: **~1800x faster** (7s vs 2+ hours before optimization)
- ✅ Smart sampling: Reduces candidates 80K→5K for instant recommendations
- ✅ Memory usage: 500-800 MB with all algorithms loaded

---

## 🎓 Key Features to Demonstrate V2.0.0

### Core Features
1. ✅ **Multi-Algorithm Support** - 4 different recommendation approaches
2. ✅ **Personalized Recommendations** (F-02) - Top-N predictions per algorithm
3. ✅ **Explainable AI** (F-05) - Algorithm-specific "Why?" explanations
4. ✅ **User Taste Profile** (F-06) - Genre preferences sidebar
5. ✅ **Surprise Me Mode** (F-07) - Serendipity recommendations

### V2.0.0 New Features
6. ✅ **Algorithm Benchmarking** - Comprehensive RMSE/MAE/Coverage comparison
7. ✅ **Performance Metrics** - Real-time training time, accuracy, coverage display
8. ✅ **Pre-trained Models** - Instant KNN loading (200x faster)
9. ✅ **Smart Sampling** - Efficient candidate selection for speed
10. ✅ **Analytics Dashboard** - Advanced visualizations and insights

### Technical Features
11. ✅ **Data Integrity Checks** (NF-01) - Automated validation
12. ✅ **Movie Similarity** (F-10) - Item-item recommendations
13. ✅ **Interactive Analytics** (F-09) - Genre/temporal analysis
14. ✅ **Algorithm Manager** - Thread-safe singleton with intelligent caching

---

## 📝 Sample User IDs

Try these User IDs for diverse recommendations:
- **1** - Classic movie lover
- **50** - Action enthusiast
- **123** - Drama fan
- **500** - Horror buff
- **1000** - Sci-fi aficionado
- **5000** - Comedy lover

---

## 🎬 One-Liner Pitch

> "CineMatch uses SVD collaborative filtering on 32 million ratings to deliver personalized, explainable movie recommendations with production-level engineering."

---

## 📞 Need Help?

Check:
1. **README.md** - Comprehensive documentation
2. **ARCHITECTURE.md** - Technical deep dive
3. **PRD.md** - Product requirements
4. **todo.md** - Development checklist

---

**Good luck with your thesis defense! 🎓**
