@echo off
echo 🎬 CineMatch V2.0 Multi-Algorithm Launcher
echo =========================================
echo.

echo ✅ Launching CineMatch V2.0 Enhanced Interface...
echo.

echo 🚀 Starting multi-algorithm recommendation system...
echo 📊 Available Algorithms:
echo    - SVD Matrix Factorization
echo    - KNN User-Based Collaborative Filtering  
echo    - KNN Item-Based Content Filtering
echo    - Hybrid (Best of All) ⭐
echo.

echo 🌐 Opening in browser: http://localhost:8502
echo 🛑 Press Ctrl+C to stop the server
echo.

REM Launch the V2.0 enhanced interface
streamlit run "app/pages/2_🎬_Recommend_V2.py" --server.port=8502 --server.address=localhost --browser.gatherUsageStats=false

pause