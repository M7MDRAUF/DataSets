@echo off
REM CineMatch V1.0.0 - Quick Start Script
REM Windows Batch File for Easy Application Launch

echo ========================================
echo CineMatch V1.0.0 - Movie Recommender
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if model exists
if not exist "models\svd_model.pkl" (
    if not exist "models\svd_model_sklearn.pkl" (
        echo.
        echo [WARNING] No trained model found!
        echo Please train the model first:
        echo   python src\model_training.py
        echo.
        pause
        exit /b 1
    )
)

REM Check if dataset exists
if not exist "data\ml-32m\ratings.csv" (
    echo.
    echo [ERROR] Dataset not found!
    echo Please download MovieLens 32M dataset to data\ml-32m\
    echo.
    pause
    exit /b 1
)

echo [2/3] Checking dependencies...
python -c "import streamlit, pandas, numpy, plotly" 2>nul
if errorlevel 1 (
    echo [ERROR] Dependencies missing! Run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo [3/3] Launching CineMatch application...
echo.
echo ========================================
echo Application will open at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Launch Streamlit
streamlit run app\main.py

pause
