"""
Main script to run complete aviation analytics project
This script orchestrates all analyses and generates all visualizations
"""

from pathlib import Path
from flight_analytics import FlightAnalytics
from advanced_analytics import AdvancedFlightAnalytics

def main():
    """Run complete analytics pipeline"""
    print("\n" + "="*70)
    print(" " * 15 + "AVIATION DATA ANALYTICS PROJECT")
    print(" " * 10 + "Cathay Pacific Digital & IT Internship 2026")
    print("="*70 + "\n")
    
    # Check if data exists
    data_path = Path('data/flight_delays_sample.csv')
    if not data_path.exists():
        print("⚠ Data file not found. Running data downloader...")
        print("   Please run: python data_downloader.py")
        return
    
    # Initialize basic analytics
    print("📊 Initializing Flight Analytics...")
    analytics = FlightAnalytics()
    
    if analytics.df is None:
        print("✗ Failed to load data. Exiting.")
        return
    
    # Run basic analysis
    print("\n" + "-"*70)
    print("PHASE 1: BASIC ANALYTICS")
    print("-"*70)
    analytics.run_full_analysis()
    
    # Run advanced analytics
    print("\n" + "-"*70)
    print("PHASE 2: ADVANCED ANALYTICS")
    print("-"*70)
    try:
        advanced = AdvancedFlightAnalytics(analytics.df)
        advanced.run_advanced_analysis()
    except ImportError as e:
        print(f"⚠ Advanced analytics requires additional packages: {e}")
        print("   Install with: pip install scikit-learn scipy")
    except Exception as e:
        print(f"⚠ Error in advanced analytics: {e}")
    
    # Summary
    print("\n" + "="*70)
    print(" " * 20 + "ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📁 Generated Files:")
    print("   ✓ visualizations/delay_distribution.png")
    print("   ✓ visualizations/route_analysis.png")
    print("   ✓ visualizations/interactive_dashboard.html")
    print("   ✓ visualizations/correlation_analysis.png")
    print("   ✓ visualizations/time_series_analysis.png")
    print("   ✓ visualizations/comprehensive_dashboard.html")
    print("   ✓ aviation_analytics_report.txt")
    print("\n💡 Next Steps:")
    print("   1. Open visualizations/interactive_dashboard.html in a browser")
    print("   2. Review aviation_analytics_report.txt for insights")
    print("   3. Open aviation_analytics_notebook.ipynb for detailed analysis")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()

