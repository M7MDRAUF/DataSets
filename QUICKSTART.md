# 🚀 CineMatch Quick Setup Guide

This guide will help you get CineMatch up and running in under 15 minutes!

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

### Step 3: Train Model & Run (5-30 minutes)

#### Option A: Quick Test (1M sample, ~2 minutes)
```powershell
python src/model_training.py
# Choose "sample" when prompted
```

#### Option B: Full Model (32M dataset, ~30 minutes)
```powershell
python src/model_training.py
# Choose "full" when prompted
```

#### Run the App
```powershell
streamlit run app/main.py
```

**Open**: http://localhost:8501

---

## 🐳 Docker Alternative (Recommended for Demo)

```powershell
# After downloading dataset and training model
docker-compose up --build
```

---

## ✅ Verification Checklist

Before running, ensure:

- [ ] Python 3.9+ installed (`python --version`)
- [ ] All dependencies installed (`pip list | grep streamlit`)
- [ ] Dataset files in `data/ml-32m/` (4 CSV files)
- [ ] Model trained (`models/svd_model.pkl` exists)
- [ ] Port 8501 available

---

## 🎯 Demo Flow (For Professor)

### 1. Home Page (30 seconds)
- Shows dataset overview
- 32M ratings, 50K+ movies
- Genre distribution chart

### 2. Recommend Page (2 minutes)
- Enter User ID: **123**
- Click **"Get Recommendations"**
- View personalized movie list
- Click **"Explain"** on first movie
- Show taste profile sidebar
- Try **"Surprise Me"** button

### 3. Analytics Page (1 minute)
- Show genre analysis
- Demonstrate movie similarity
- Search for **"Matrix"**
- Find similar movies

### 4. Technical Highlights
- Mention RMSE < 0.87 ✅
- Response time < 2 seconds ✅
- Explainable AI feature ✅
- Docker containerization ✅

---

## 🔧 Troubleshooting

### "Data Integrity Check Failed"
**Solution**: Download dataset and place in `data/ml-32m/`

### "Model file not found"
**Solution**: Run `python src/model_training.py`

### "Import streamlit could not be resolved"
**Solution**: 
```powershell
pip install -r requirements.txt
```

### "Memory Error during training"
**Solution**: Choose "sample" option (1M ratings)

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Model RMSE | < 0.87 ✅ |
| Training Time (full) | 15-30 min |
| Training Time (sample) | 1-2 min |
| Recommendation Time | < 2 sec |
| Memory Usage | 6-8 GB |

---

## 🎓 Key Features to Demonstrate

1. ✅ **Personalized Recommendations** (F-02)
2. ✅ **Explainable AI** (F-05) - "Why?" button
3. ✅ **User Taste Profile** (F-06) - Sidebar
4. ✅ **Surprise Me Mode** (F-07)
5. ✅ **Data Integrity Checks** (NF-01)
6. ✅ **Movie Similarity** (F-10)
7. ✅ **Interactive Analytics** (F-09)

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
