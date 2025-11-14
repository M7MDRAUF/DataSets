# CineMatch V2.1.0 - Troubleshooting Guide

## 📋 Overview

Comprehensive troubleshooting guide for common issues in CineMatch. This document covers setup problems, runtime errors, performance issues, and deployment challenges.

---

## 🚨 Most Common Issues

### 1. Git LFS Models Not Loaded

**Symptom**:
```
FileNotFoundError: models/user_knn_model.pkl not found
OR
RuntimeError: File is a Git LFS pointer, not actual model
OR
Model file is only 133 bytes (should be 266MB)
```

**Cause**: Git LFS not installed or models not pulled.

**Solution**:

```powershell
# Step 1: Install Git LFS (if not installed)
# Windows (with Chocolatey)
choco install git-lfs

# macOS
brew install git-lfs

# Linux (Debian/Ubuntu)
sudo apt-get install git-lfs

# Step 2: Initialize Git LFS in repository
git lfs install

# Step 3: Pull actual model files
git lfs pull

# Step 4: Verify model sizes
ls -lh models/
# Expected output:
#   user_knn_model.pkl    ~266MB
#   item_knn_model.pkl    ~260MB
#   content_based_model.pkl ~300MB
```

**Verification**:

```powershell
# Check if file is LFS pointer
Get-Content models/user_knn_model.pkl | Select-Object -First 3
# If you see "version https://git-lfs.github.com/spec/v1", it's a pointer

# After git lfs pull, you should see binary data (unreadable)
```

**Prevention**:
- Always run `git lfs install` after cloning
- Add to documentation: "Run `git lfs pull` before starting app"

---

### 2. Port 8501 Already in Use

**Symptom**:
```
OSError: [Errno 98] Address already in use
OR
streamlit.errors.StreamlitAddressInUseError: Can't bind to port 8501
```

**Cause**: Another Streamlit app or process using port 8501.

**Solution**:

```powershell
# Option 1: Kill process using port 8501
# Windows
netstat -ano | findstr :8501
# Find PID, then:
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :8501
kill -9 <PID>

# Option 2: Use different port
streamlit run app/main.py --server.port=8502

# Option 3: Let Streamlit auto-assign port
streamlit run app/main.py --server.port=0
```

**Prevention**:
- Stop Streamlit apps before starting new ones
- Use unique ports for different projects
- Add `--server.port` to launch scripts

---

### 3. Dataset Files Not Found (NF-01 Failure)

**Symptom**:
```
❌ DATA INTEGRITY CHECK FAILED
Missing files: data/ml-32m/ratings.csv, data/ml-32m/movies.csv
```

**Cause**: MovieLens 32M dataset not downloaded.

**Solution**:

```powershell
# Step 1: Download MovieLens 32M dataset
# Visit: http://grouplens.org/datasets/movielens/32m/
# Or direct link: http://files.grouplens.org/datasets/movielens/ml-32m.zip

# Step 2: Extract to correct location
# Windows (PowerShell)
Expand-Archive -Path ml-32m.zip -DestinationPath data/

# Linux/macOS
unzip ml-32m.zip -d data/

# Step 3: Verify structure
tree data/ml-32m
# Expected:
# data/ml-32m/
#   ├── ratings.csv (32M ratings, ~600MB)
#   ├── movies.csv
#   ├── tags.csv
#   ├── links.csv
#   └── README.txt

# Step 4: Restart application
streamlit run app/main.py
```

**Verification**:

```powershell
# Check file sizes
ls -lh data/ml-32m/
# ratings.csv should be ~600-700MB
```

---

### 4. Memory Errors with Large Models

**Symptom**:
```
MemoryError: Unable to allocate array with shape (330000, 87000)
OR
Killed (process terminated by OS)
```

**Cause**: Insufficient RAM for loading all models simultaneously.

**Solution**:

```python
# Option 1: Load algorithms on-demand (recommended)
manager = AlgorithmManager()
# Don't load all at once:
# recommender1 = manager.get_algorithm("User-KNN")  # 300MB
# recommender2 = manager.get_algorithm("Item-KNN")  # 300MB
# recommender3 = manager.get_algorithm("Content-Based")  # 400MB
# Total: 1GB+ ❌

# Instead, load one at a time:
recommender = manager.get_algorithm("User-KNN")  # Only 300MB ✅

# Option 2: Use smaller algorithms
recommender = manager.get_algorithm("SVD")  # Only 50MB

# Option 3: Increase system swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**System Requirements**:
- **Minimum**: 8GB RAM (can run 1-2 algorithms)
- **Recommended**: 16GB RAM (can run all 5 algorithms)
- **Optimal**: 32GB RAM (smooth multi-algorithm benchmarking)

---

### 5. Slow Recommendations (> 10 seconds)

**Symptom**:
- Recommendations take 30-60 seconds to generate
- Application feels sluggish
- High CPU usage during recommendation

**Cause**: Not using pre-trained models OR smart sampling disabled.

**Solution**:

```python
# ✅ Correct: Use pre-trained models (fast)
recommender = manager.get_algorithm("User-KNN")  # Loads 266MB model
recommendations = recommender.recommend(user_id=123, n=10)
# Expected: ~1s

# ❌ Wrong: Training from scratch (slow)
recommender = UserKNNRecommender()
recommender.fit(ratings_df)  # 2-4 hours!
recommendations = recommender.recommend(user_id=123, n=10)

# Ensure smart sampling is enabled in user_knn_recommender.py:
# candidate_sample_size = min(5000, len(candidate_movies))  # ✅
# NOT: candidate_sample_size = len(candidate_movies)  # ❌ (80K movies)
```

**Performance Targets**:
- SVD: ~2s
- User-KNN: ~1s (with smart sampling)
- Item-KNN: ~1s
- Content-Based: ~0.5s
- Hybrid: ~3s

If slower, check:
1. Are pre-trained models loaded?
2. Is smart sampling enabled?
3. Is Streamlit caching working? (check `@st.cache_resource`)

---

### 6. Docker Container Fails to Start

**Symptom**:
```
docker: Error response from daemon: failed to create shim
OR
Container exits immediately with code 1
```

**Common Causes & Solutions**:

**A. Git LFS Models Not in Image**

```dockerfile
# Solution: Run git lfs pull BEFORE building Docker image
git lfs pull

# Then build:
docker build -t cinematch:v2.1.0 .
```

**B. Missing Dataset**

```dockerfile
# Solution: Mount dataset as volume
docker run -v $(pwd)/data:/app/data cinematch:v2.1.0

# OR add COPY command in Dockerfile:
COPY data/ml-32m /app/data/ml-32m
```

**C. Port Already in Use**

```bash
# Solution: Use different host port
docker run -p 8502:8501 cinematch:v2.1.0
```

**Debugging**:

```bash
# Check container logs
docker logs <container_id>

# Run interactively to see errors
docker run -it --entrypoint /bin/bash cinematch:v2.1.0

# Inside container, verify files:
ls -lh models/
ls -lh data/ml-32m/
```

---

### 7. Streamlit "Connection Error" or White Screen

**Symptom**:
- Browser shows "Connection error" message
- White screen with spinning loader
- "Please wait..." never completes

**Causes & Solutions**:

**A. Streamlit Server Not Running**

```bash
# Check if process is running
ps aux | grep streamlit  # Linux/macOS
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"}  # Windows

# Restart if needed
streamlit run app/main.py
```

**B. Firewall Blocking Port**

```bash
# Allow port 8501 through firewall
# Windows
netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8501

# Linux (ufw)
sudo ufw allow 8501
```

**C. Browser Cache Issues**

```
# Solution 1: Hard refresh
Ctrl+Shift+R (Windows/Linux)
Cmd+Shift+R (macOS)

# Solution 2: Clear browser cache
# Or use incognito/private mode
```

**D. Streamlit Configuration Issues**

```toml
# Create .streamlit/config.toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

---

## 🐛 Runtime Errors

### 8. KeyError: 'movieId' or 'userId'

**Symptom**:
```python
KeyError: 'movieId'
```

**Cause**: DataFrame column name mismatch (e.g., `movie_id` vs `movieId`).

**Solution**:

```python
# Check column names
print(ratings_df.columns.tolist())
# Should be: ['userId', 'movieId', 'rating', 'timestamp']

# If different, standardize:
ratings_df = ratings_df.rename(columns={
    'user_id': 'userId',
    'movie_id': 'movieId'
})
```

---

### 9. ValueError: Unknown Algorithm Name

**Symptom**:
```python
ValueError: Algorithm 'content-based' not recognized
```

**Cause**: Incorrect algorithm name (case-sensitive, hyphenation matters).

**Solution**:

```python
# ✅ Correct algorithm names (use these):
valid_algorithms = [
    "SVD",
    "User-KNN",      # Note: hyphen, not underscore
    "Item-KNN",
    "Content-Based", # Note: capital letters
    "Hybrid"
]

# ❌ Wrong:
# "content-based" (lowercase)
# "User_KNN" (underscore)
# "user-knn" (lowercase)
```

---

### 10. Prediction Returns NaN

**Symptom**:
```python
predicted_rating = recommender.predict(user_id=123, movie_id=456)
# Result: nan
```

**Cause**: User or movie not in training data (cold-start problem).

**Solution**:

```python
# Use Content-Based for new users (no cold-start)
recommender = manager.get_algorithm("Content-Based")

# OR handle NaN predictions:
predicted_rating = recommender.predict(user_id=123, movie_id=456)
if pd.isna(predicted_rating):
    predicted_rating = global_mean  # Fallback to average rating
```

---

## ⚡ Performance Issues

### 11. Streamlit App Slow to Load

**Symptom**: First page load takes > 30 seconds.

**Optimization Checklist**:

```python
# 1. Ensure caching is enabled
@st.cache_resource
def load_manager():
    return AlgorithmManager()

@st.cache_data
def load_datasets():
    return load_ratings(), load_movies()

# 2. Load models on-demand, not at startup
# Don't do this in main.py:
# manager.get_algorithm("User-KNN")  # Loads 266MB ❌

# 3. Sample ratings for UI (don't load all 32M)
ratings_df = load_ratings(sample_size=100000)  # ✅

# 4. Check debug mode is off
streamlit run app/main.py --server.runOnSave=false
```

---

### 12. Analytics Dashboard Timeout

**Symptom**: "Script exceeded time limit" when loading Analytics page.

**Cause**: Calculating metrics for all 5 algorithms takes too long.

**Solution**:

```python
# In pages/3_Analytics.py:
# Use emergency-optimized metrics calculation
metrics_df = manager.get_all_algorithm_metrics()
# This uses stratified sampling (7s vs 2+ hours)

# If still slow, cache results:
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_metrics():
    return manager.get_all_algorithm_metrics()
```

---

## 🚀 Deployment Issues

### 13. Streamlit Cloud Deployment Fails

**Common Issues**:

**A. requirements.txt Missing Dependencies**

```txt
# Ensure all dependencies listed:
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.5.0
joblib>=1.3.0
plotly>=5.14.0
```

**B. Git LFS Models Not in Cloud**

```bash
# Streamlit Cloud supports Git LFS
# Ensure .gitattributes exists:
*.pkl filter=lfs diff=lfs merge=lfs -text

# Push models to LFS:
git lfs push --all origin main
```

**C. Resource Limits Exceeded**

```python
# Streamlit Cloud has 1GB RAM limit
# Solution: Use SVD or Content-Based (smaller models)
# Avoid loading all 5 algorithms simultaneously
```

---

### 14. Heroku Deployment "Slug Size Too Large"

**Symptom**:
```
Compiled slug size: 1.2GB is too large (max is 500MB)
```

**Cause**: Models (~800MB) exceed Heroku slug size limit.

**Solutions**:

```bash
# Option 1: Use Heroku Large Dynos (paid plan)

# Option 2: Load models from external storage
# - Upload models to S3/GCS
# - Download at runtime

# Option 3: Use Docker deployment instead of Heroku
# - AWS ECS, Google Cloud Run, Azure Container Instances
```

---

### 15. Docker Volume Permissions Error

**Symptom**:
```
PermissionError: [Errno 13] Permission denied: '/app/data/ml-32m/ratings.csv'
```

**Cause**: File ownership mismatch between host and container.

**Solution**:

```dockerfile
# In Dockerfile, set correct user:
RUN useradd -m -u 1000 appuser
USER appuser

# OR run container with host user:
docker run --user $(id -u):$(id -g) -v $(pwd)/data:/app/data cinematch
```

---

## 🔍 Debugging Tips

### Enable Debug Logging

```python
# Add to top of main.py:
import logging
logging.basicConfig(level=logging.DEBUG)

# For Streamlit:
streamlit run app/main.py --logger.level=debug
```

### Check System Resources

```bash
# Memory usage
free -h  # Linux
vm_stat  # macOS
Get-ComputerInfo | Select-Object CsFreePhysicalMemory  # Windows

# CPU usage
top  # Linux/macOS
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10  # Windows

# Disk space
df -h  # Linux/macOS
Get-PSDrive  # Windows
```

### Profile Performance

```python
import time

# Time algorithm loading
start = time.time()
recommender = manager.get_algorithm("User-KNN")
print(f"Load time: {time.time() - start:.2f}s")
# Expected: ~1.5s

# Time recommendations
start = time.time()
recommendations = recommender.recommend(user_id=123, n=10)
print(f"Recommendation time: {time.time() - start:.2f}s")
# Expected: ~1s
```

---

## 📞 Getting Help

### Before Opening an Issue

1. **Check this guide** for common solutions
2. **Verify system requirements**:
   - Python 3.9-3.13
   - 8GB+ RAM
   - Git LFS installed
3. **Check logs**:
   ```bash
   streamlit run app/main.py 2>&1 | tee debug.log
   ```
4. **Test with minimal example**:
   ```python
   from src.algorithms.algorithm_manager import AlgorithmManager
   manager = AlgorithmManager()
   recommender = manager.get_algorithm("SVD")  # Smallest model
   ```

### Issue Template

When reporting issues, include:

```
**Environment**:
- OS: Windows 11 / macOS 14 / Ubuntu 22.04
- Python version: 3.11.5
- Streamlit version: 1.32.0
- RAM: 16GB
- Git LFS installed: Yes/No

**Issue**:
[Describe the problem]

**Steps to Reproduce**:
1. Run `streamlit run app/main.py`
2. Select "User-KNN" algorithm
3. Error occurs

**Error Message**:
```
[Paste complete error stack trace]
```

**What I Tried**:
- Ran git lfs pull ✅
- Checked dataset files ✅
- Still seeing error ❌
```

---

## 🔧 Quick Fix Checklist

Before troubleshooting, verify:

- [ ] Git LFS installed: `git lfs version`
- [ ] Models pulled: `git lfs pull` (check file sizes)
- [ ] Dataset downloaded: `ls -lh data/ml-32m/ratings.csv` (~600MB)
- [ ] Port 8501 available: `netstat -ano | findstr :8501`
- [ ] Python 3.9-3.13: `python --version`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Sufficient RAM: At least 8GB free
- [ ] Streamlit up-to-date: `pip install --upgrade streamlit`

---

## 📚 Additional Resources

- **Documentation**: See `README.md`, `QUICKSTART.md`, `ARCHITECTURE.md`
- **API Reference**: See `API_DOCUMENTATION.md`
- **Module Docs**: See `MODULE_DOCUMENTATION.md`
- **Git LFS Guide**: https://git-lfs.github.com/
- **Streamlit Docs**: https://docs.streamlit.io/

---

*Document Version: 1.0*
*Last Updated: November 13, 2025*
*Part of CineMatch V2.1.0 Documentation Suite*
