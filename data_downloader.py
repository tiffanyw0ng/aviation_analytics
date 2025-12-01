"""
Data Downloader for Aviation Analytics Project
Downloads real aviation data from public sources for analysis
"""

import requests
import pandas as pd
import os
from pathlib import Path

def download_bts_flight_data(year=2023, month=1):
    """
    Download flight delay data from Bureau of Transportation Statistics
    Note: This is a template - actual BTS data requires specific API calls or manual download
    """
    print(f"Downloading BTS flight data for {year}-{month:02d}...")
    
    # BTS data URL structure (example)
    # Actual URL would be: https://www.transtats.bts.gov/DL_SelectFields.aspx?Table_ID=236
    base_url = "https://www.transtats.bts.gov/DL_SelectFields.aspx"
    
    print("Note: BTS data requires manual download from:")
    print("https://www.transtats.bts.gov/Tables.asp?DB_ID=120")
    print("\nFor this demo, we'll use a sample dataset generator.")
    
    return None

def download_openflights_data():
    """Download airport and route data from OpenFlights"""
    print("Downloading OpenFlights data...")
    
    urls = {
        'airports': 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat',
        'routes': 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat',
        'airlines': 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat'
    }
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    downloaded_files = {}
    
    for name, url in urls.items():
        try:
            print(f"  Downloading {name}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filepath = data_dir / f'{name}.csv'
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            downloaded_files[name] = filepath
            print(f"  ✓ Saved to {filepath}")
        except Exception as e:
            print(f"  ✗ Error downloading {name}: {e}")
    
    return downloaded_files

def create_sample_flight_data():
    """Create realistic sample flight data for demonstration"""
    import numpy as np
    from datetime import datetime, timedelta
    
    print("Creating sample flight delay data...")
    
    np.random.seed(42)
    n_flights = 5000
    
    # Generate sample data
    airlines = ['CX', 'KA', 'AA', 'UA', 'DL', 'LH', 'BA', 'QF', 'SQ', 'EK']
    airports = ['HKG', 'JFK', 'LAX', 'LHR', 'NRT', 'SYD', 'DXB', 'SIN', 'FRA', 'CDG']
    
    data = {
        'date': pd.date_range(start='2024-01-01', periods=n_flights, freq='h'),
        'airline': np.random.choice(airlines, n_flights),
        'origin': np.random.choice(airports, n_flights),
        'destination': np.random.choice(airports, n_flights),
        'scheduled_departure': np.random.randint(0, 24, n_flights) * 100 + np.random.randint(0, 60, n_flights),
        'actual_departure': None,
        'scheduled_arrival': None,
        'actual_arrival': None,
        'departure_delay': None,
        'arrival_delay': None,
        'flight_duration': None,
        'distance': None
    }
    
    df = pd.DataFrame(data)
    
    # Calculate realistic delays and times
    df['departure_delay'] = np.random.normal(10, 30, n_flights).astype(int)
    df['departure_delay'] = df['departure_delay'].clip(-20, 300)  # Realistic delay range
    
    df['actual_departure'] = df['scheduled_departure'] + df['departure_delay']
    
    # Flight duration (hours) based on distance
    df['distance'] = np.random.normal(2000, 1000, n_flights).astype(int)
    df['distance'] = df['distance'].clip(500, 8000)
    df['flight_duration'] = (df['distance'] / 800).round(1)  # Average speed ~800 km/h
    
    # Arrival times
    df['scheduled_arrival'] = (df['scheduled_departure'] + df['flight_duration'] * 100).astype(int)
    df['arrival_delay'] = df['departure_delay'] + np.random.normal(0, 10, n_flights).astype(int)
    df['actual_arrival'] = (df['scheduled_arrival'] + df['arrival_delay']).astype(int)
    
    # Add some categorical features
    df['delay_category'] = pd.cut(df['departure_delay'], 
                                   bins=[-np.inf, 0, 15, 60, np.inf],
                                   labels=['Early/On-time', 'Minor Delay', 'Moderate Delay', 'Major Delay'])
    
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['hour'] = df['date'].dt.hour
    
    # Save to CSV
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / 'flight_delays_sample.csv'
    df.to_csv(filepath, index=False)
    print(f"✓ Sample data saved to {filepath}")
    
    return filepath

if __name__ == "__main__":
    print("=" * 60)
    print("Aviation Data Downloader")
    print("=" * 60)
    print()
    
    # Download OpenFlights data
    openflights_files = download_openflights_data()
    print()
    
    # Create sample flight delay data
    sample_file = create_sample_flight_data()
    print()
    
    print("=" * 60)
    print("Data download complete!")
    print("=" * 60)
    print("\nFiles ready for analysis:")
    if openflights_files:
        for name, path in openflights_files.items():
            print(f"  - {name}: {path}")
    print(f"  - flight_delays_sample.csv: {sample_file}")

