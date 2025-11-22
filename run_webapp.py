"""
Tesla Stock Predictor - Streamlit App Launcher
Double-click this file to launch the web app!
"""

import subprocess
import sys
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the Streamlit app
    app_path = os.path.join(script_dir, 'dashboard', 'app.py')
    
    # Check if app exists
    if not os.path.exists(app_path):
        print(f"Error: App not found at {app_path}")
        input("Press Enter to exit...")
        return
    
    print("="*60)
    print("  🚀 TESLA STOCK PREDICTOR - WEB APP")
    print("="*60)
    print()
    print("Starting Streamlit server...")
    print("The app will open in your web browser automatically.")
    print()
    print("To stop the server, press Ctrl+C or close this window.")
    print("="*60)
    print()
    
    # Change to script directory
    os.chdir(script_dir)
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            app_path,
            '--server.headless=false',
            '--browser.gatherUsageStats=false'
        ])
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Streamlit is installed: pip install streamlit")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
