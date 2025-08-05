#!/usr/bin/env python3
"""
Dashboard Dependency Test
Quick test script to verify dashboard dependencies and data availability.

This script checks if all required components are available for the
Streamlit dashboard to run successfully.

Author: AI Assistant
Created: 2025
"""

import sys
from pathlib import Path

def test_dependencies():
    """Test if all required dependencies are available."""
    print("🔧 Testing Dashboard Dependencies")
    print("=" * 40)
    
    dependencies = [
        ('streamlit', 'Interactive web framework'),
        ('plotly', 'Advanced visualizations'),
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing')
    ]
    
    available = []
    missing = []
    
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"✅ {package:12} - {description}")
            available.append(package)
        except ImportError:
            print(f"❌ {package:12} - {description} (MISSING)")
            missing.append(package)
    
    print(f"\n📊 Status: {len(available)}/{len(dependencies)} dependencies available")
    
    if missing:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing)}")
        print("\n💡 Installation commands:")
        print(f"   pip install {' '.join(missing)}")
        print("   # OR")
        print(f"   conda install {' '.join(missing)}")
        return False
    else:
        print("\n🎉 All dependencies available!")
        return True

def test_data_files():
    """Test if required data files exist."""
    print("\n📁 Testing Data Files")
    print("=" * 40)
    
    required_files = [
        ('pathfinder/output/full_metrics_raw.csv', 'Advanced metrics data'),
        ('career_path_scores.csv', 'Basic scoring data'),
        ('pathfinder_dashboard.html', 'Standalone dashboard')
    ]
    
    available_files = []
    missing_files = []
    
    for filepath, description in required_files:
        if Path(filepath).exists():
            file_size = Path(filepath).stat().st_size
            print(f"✅ {filepath:35} - {description} ({file_size:,} bytes)")
            available_files.append(filepath)
        else:
            print(f"❌ {filepath:35} - {description} (MISSING)")
            missing_files.append(filepath)
    
    print(f"\n📊 Status: {len(available_files)}/{len(required_files)} data files available")
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        print("\n💡 Generation commands:")
        for filepath in missing_files:
            if 'full_metrics_raw.csv' in filepath:
                print("   python3 pathfinder/metrics_engine.py")
            elif 'career_path_scores.csv' in filepath:
                print("   python3 score_career_paths.py")
            elif 'pathfinder_dashboard.html' in filepath:
                print("   python3 pathfinder_dashboard_standalone.py")
        return False
    else:
        print("\n🎉 All data files available!")
        return True

def test_dashboard_launch():
    """Test if the dashboard can be launched."""
    print("\n🚀 Testing Dashboard Launch")
    print("=" * 40)
    
    try:
        # Import required modules
        import streamlit as st
        import plotly.express as px
        import pandas as pd
        
        # Check data loading
        metrics_file = Path("pathfinder/output/full_metrics_raw.csv")
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            print(f"✅ Data loaded successfully: {len(df)} career paths, {len(df.columns)} metrics")
            
            # Test basic calculations
            m_cols = [col for col in df.columns if col.startswith('M')]
            c_cols = [col for col in df.columns if col.startswith('C')]
            print(f"✅ Found {len(m_cols)} M-series metrics")
            print(f"✅ Found {len(c_cols)} C-series metrics")
            
            print("\n🎉 Dashboard ready to launch!")
            print("\n🚀 Launch command:")
            print("   streamlit run pathfinder/dashboard.py")
            return True
        else:
            print("❌ Metrics data file not found")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function."""
    print("🧭 Advanced Pathfinder Dashboard - Dependency Test")
    print("=" * 55)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test data files
    data_ok = test_data_files()
    
    # Test dashboard if dependencies are available
    if deps_ok and data_ok:
        launch_ok = test_dashboard_launch()
    else:
        launch_ok = False
    
    # Summary
    print("\n" + "=" * 55)
    print("📊 SUMMARY")
    print("=" * 55)
    
    if deps_ok and data_ok and launch_ok:
        print("🎉 ✅ Dashboard is ready to launch!")
        print("\n🚀 Next steps:")
        print("   1. streamlit run pathfinder/dashboard.py")
        print("   2. Open browser to http://localhost:8501")
        print("   3. Adjust weights and explore careers")
        print("\n💡 Alternative options:")
        print("   • HTML Dashboard: open pathfinder_dashboard.html")
        print("   • CSV Analysis: open pathfinder/output/full_metrics_raw.csv")
        return 0
    else:
        print("⚠️ ❌ Dashboard setup incomplete")
        print("\n🔧 Required actions:")
        if not deps_ok:
            print("   1. Install missing dependencies (see above)")
        if not data_ok:
            print("   2. Generate missing data files (see above)")
        
        print("\n💡 Alternative options:")
        print("   • Use existing HTML dashboard: pathfinder_dashboard.html")
        print("   • Use basic scoring: python3 score_career_paths.py")
        return 1

if __name__ == "__main__":
    exit(main())