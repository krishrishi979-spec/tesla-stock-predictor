@echo off
echo ============================================================
echo   TESLA STOCK PREDICTOR - WEB APP
echo ============================================================
echo.
echo Starting Streamlit server...
echo The app will open in your web browser automatically.
echo.
echo To stop the server, press Ctrl+C or close this window.
echo ============================================================
echo.

cd /d "%~dp0"
streamlit run dashboard/app.py --server.headless=false --browser.gatherUsageStats=false

pause
