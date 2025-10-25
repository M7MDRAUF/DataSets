# CineMatch V1.0.0 - Scripts & Automation

This directory contains automation scripts for testing, validation, and demo preparation.

---

## 📋 Available Scripts

### 1. `validate_system.py` - Pre-Launch System Validation
**Purpose**: Comprehensive validation of entire system before thesis defense

**What it checks:**
- ✅ Project structure (all directories exist)
- ✅ Dataset files (all 4 CSVs present and correct size)
- ✅ Python dependencies (all required packages installed)
- ✅ Source code files (all modules present)
- ✅ Trained model (svd_model.pkl or svd_model_sklearn.pkl)
- ✅ Documentation (README, QUICKSTART, etc.)
- ✅ Docker configuration (Dockerfile, docker-compose.yml)
- ✅ Engine import test (can load recommendation engine)
- ✅ Data loading test (can read dataset)

**Usage:**
```bash
python scripts/validate_system.py
```

**Expected Output:**
```
CineMatch V1.0.0 - System Validation
====================================================================
✅ Project Structure
✅ Dataset
✅ Dependencies
✅ Source Code
✅ Model
✅ Documentation
✅ Docker
✅ Scripts
✅ Engine Import
✅ Data Loading

Score: 10/10 checks passed (100.0%)

🎉 ALL VALIDATIONS PASSED!
✅ System is ready for thesis defense!
```

---

### 2. `test_features.py` - Automated Feature Testing
**Purpose**: Test all 10 PRD features automatically with detailed reporting

**What it tests:**
- F-01: Personalized Recommendations
- F-02: SVD Collaborative Filtering
- F-03: Top-N Recommendations
- F-04: Accuracy Metrics (RMSE)
- F-05: Explainable AI
- F-06: User Taste Profiling
- F-07: "Surprise Me" Feature
- F-08: Feedback Collection
- F-09: Data Visualizations
- F-10: Movie Similarity

**Usage:**
```bash
python scripts/test_features.py
```

**Expected Output:**
```
CineMatch V1.0.0 - Feature Testing Suite
====================================================================
Testing F-01: Personalized Recommendations...
Testing F-02: Collaborative Filtering (SVD)...
Testing F-03: Top-N Recommendations...
...

Test Results Summary
====================================================================
✅ PASS F-01: Personalized Recommendations
    → Generated 5 recommendations for User 1

✅ PASS F-02: SVD Collaborative Filtering
    → Model generates valid predictions (range: 0-5)

...

Final Score: 10/10 tests passed (100.0%)

🎉 ALL TESTS PASSED! System is thesis-ready!
```

---

### 3. `demo_script.md` - Thesis Defense Demo Guide
**Purpose**: Step-by-step script for 5-minute thesis defense demonstration

**What it includes:**
- ✅ Pre-demo checklist
- ✅ 5-minute demo flow with timing
- ✅ What to say at each step
- ✅ Which features to demonstrate
- ✅ Q&A preparation with answers
- ✅ Backup plan if demo fails

**Usage:**
1. Open `demo_script.md`
2. Review the day before defense
3. Practice the 5-minute flow
4. Prepare answers to anticipated questions

**Demo Structure:**
- Opening (30 sec)
- Data & Model (1 min)
- Core Recommendations (2 min)
- Analytics (1 min)
- Technical Deep Dive (30 sec)
- Closing (30 sec)

---

## 🚀 Quick Start Scripts

### Windows: `start.bat`
**Purpose**: One-click application launch (Windows batch file)

**What it does:**
1. Checks if virtual environment exists
2. Activates virtual environment
3. Verifies model and dataset exist
4. Checks Python dependencies
5. Launches Streamlit application

**Usage:**
```bash
# Double-click start.bat in File Explorer
# Or run from command line:
start.bat
```

**Output:**
```
========================================
CineMatch V1.0.0 - Movie Recommender
========================================

[1/3] Activating virtual environment...
[2/3] Checking dependencies...
[3/3] Launching CineMatch application...

========================================
Application will open at:
http://localhost:8501

Press Ctrl+C to stop the server
========================================
```

---

## 📊 Testing Workflow

### Before Thesis Defense

**Run this sequence:**

```bash
# 1. Validate entire system
python scripts/validate_system.py

# 2. Test all features
python scripts/test_features.py

# 3. Launch application
start.bat
# Or: streamlit run app/main.py

# 4. Manual testing:
#    - Test with User IDs: 1, 123, 1000
#    - Try all tabs (Home, Recommend, Analytics)
#    - Test "Surprise Me" feature
#    - Check explanations display

# 5. Review demo script
# Open: scripts/demo_script.md
# Practice: 5-minute presentation flow
```

---

## 🐛 Troubleshooting

### Script Fails with "Module not found"

**Problem:** Python can't find project modules

**Solution:**
```bash
# Make sure you're in project root directory
cd C:\Users\moham\OneDrive\Documents\studying\ai_mod_project\Copilot

# Activate virtual environment
venv\Scripts\activate

# Run script
python scripts/validate_system.py
```

### Validation Script Shows Missing Model

**Problem:** No trained model found

**Solution:**
```bash
# Train model (takes 15-30 minutes)
python src/model_training_sklearn.py

# Choose 'full' for thesis, 'sample' for testing
```

### Feature Tests Fail

**Problem:** Some features not working

**Solutions:**

1. **Check model loaded:**
   ```bash
   # Verify model file exists
   dir models\
   ```

2. **Check dataset:**
   ```bash
   # Verify dataset files
   dir data\ml-32m\
   ```

3. **Check dependencies:**
   ```bash
   pip list | Select-String "streamlit|pandas|sklearn"
   ```

4. **Re-run with debug:**
   ```bash
   python scripts/test_features.py
   # Read error messages carefully
   ```

---

## 📁 Script File Locations

```
Copilot/
├── start.bat                      # Quick launch (Windows)
├── scripts/
│   ├── README.md                  # This file
│   ├── validate_system.py         # System validation
│   ├── test_features.py           # Feature testing
│   └── demo_script.md             # Thesis demo guide
```

---

## 🎯 Recommended Testing Order

### Day Before Defense

1. ✅ **Full System Check**
   ```bash
   python scripts/validate_system.py
   ```

2. ✅ **Feature Testing**
   ```bash
   python scripts/test_features.py
   ```

3. ✅ **Manual Application Test**
   ```bash
   start.bat
   # Test all features manually
   ```

4. ✅ **Demo Rehearsal**
   - Read `demo_script.md`
   - Practice 5-minute presentation
   - Prepare for Q&A

5. ✅ **Backup Screenshots**
   - Take screenshots of all features
   - Save in case live demo fails

### Day of Defense

1. ✅ **Quick Validation** (5 min before)
   ```bash
   python scripts/validate_system.py
   ```

2. ✅ **Launch Application** (keep running)
   ```bash
   start.bat
   ```

3. ✅ **Open Demo Script**
   - Have `demo_script.md` open on second screen
   - Follow the 5-minute flow

---

## 💡 Tips for Success

### Testing Tips
- ✅ Run validation script BEFORE feature tests
- ✅ Test with multiple user IDs (1, 123, 1000)
- ✅ Check both "Recommend" and "Surprise Me" features
- ✅ Verify explanations are clear and relevant
- ✅ Test all 4 tabs in Analytics page

### Demo Tips
- ✅ Practice the 5-minute flow multiple times
- ✅ Know where each feature is in the UI
- ✅ Have test user IDs ready (1, 123, 1000)
- ✅ Prepare answers to Q&A questions
- ✅ Have backup screenshots if demo fails
- ✅ Keep demo_script.md open during presentation

### Common Issues
- 🔴 **Model not found**: Run model_training_sklearn.py
- 🔴 **Dataset missing**: Download ml-32m to data/ml-32m/
- 🟡 **Slow loading**: Use sample_size in load_ratings()
- 🟡 **Memory issues**: Close other applications

---

## 📞 Support

If scripts fail or features don't work:

1. Check error messages carefully
2. Run validation script to identify issues
3. Review QUICKSTART.md for setup steps
4. Check ARCHITECTURE.md for technical details
5. Review troubleshooting section in README.md

---

**Last Updated**: October 24, 2025  
**Version**: 1.0.0  
**Status**: Ready for thesis defense
