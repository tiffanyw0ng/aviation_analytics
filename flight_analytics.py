"""
Aviation Data Analytics Project
Comprehensive analysis of flight data with visualizations
Perfect for showcasing data analytics skills for Cathay Pacific internship
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FlightAnalytics:
    """Main class for flight data analysis and visualization"""
    
    def __init__(self, data_path=None):
        """Initialize with flight data"""
        if data_path is None:
            data_path = Path('data/flight_delays_sample.csv')
        
        self.data_path = Path(data_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load flight data from CSV"""
        try:
            self.df = pd.read_csv(self.data_path)
            # Convert date column if it exists
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"✓ Loaded {len(self.df)} flight records")
            return True
        except FileNotFoundError:
            print(f"✗ Data file not found: {self.data_path}")
            print("  Please run data_downloader.py first")
            return False
    
    def overview_statistics(self):
        """Generate overview statistics"""
        if self.df is None:
            return None
        
        stats = {
            'total_flights': len(self.df),
            'total_airlines': self.df['airline'].nunique() if 'airline' in self.df.columns else 0,
            'total_airports': pd.concat([self.df['origin'], self.df['destination']]).nunique() if 'origin' in self.df.columns else 0,
            'avg_departure_delay': self.df['departure_delay'].mean() if 'departure_delay' in self.df.columns else 0,
            'on_time_percentage': (self.df['departure_delay'] <= 0).sum() / len(self.df) * 100 if 'departure_delay' in self.df.columns else 0,
            'avg_flight_duration': self.df['flight_duration'].mean() if 'flight_duration' in self.df.columns else 0,
        }
        
        return stats
    
    def visualize_delay_distribution(self, save_path='visualizations/delay_distribution.png'):
        """Visualize distribution of flight delays"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Flight Delay Analysis', fontsize=16, fontweight='bold')
        
        # 1. Delay distribution histogram
        axes[0, 0].hist(self.df['departure_delay'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.df['departure_delay'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["departure_delay"].mean():.1f} min')
        axes[0, 0].set_xlabel('Departure Delay (minutes)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Departure Delays')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Delay by airline
        if 'airline' in self.df.columns:
            airline_delays = self.df.groupby('airline')['departure_delay'].mean().sort_values()
            axes[0, 1].barh(airline_delays.index, airline_delays.values, color='steelblue')
            axes[0, 1].set_xlabel('Average Delay (minutes)')
            axes[0, 1].set_title('Average Delay by Airline')
            axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. Delay category pie chart
        if 'delay_category' in self.df.columns:
            delay_counts = self.df['delay_category'].value_counts()
            axes[1, 0].pie(delay_counts.values, labels=delay_counts.index, autopct='%1.1f%%',
                          startangle=90, colors=sns.color_palette("Set2"))
            axes[1, 0].set_title('Delay Category Distribution')
        
        # 4. Delay by day of week
        if 'day_of_week' in self.df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_delays = self.df.groupby('day_of_week')['departure_delay'].mean()
            day_delays = day_delays.reindex([d for d in day_order if d in day_delays.index])
            axes[1, 1].plot(day_delays.index, day_delays.values, marker='o', linewidth=2, markersize=8)
            axes[1, 1].set_xlabel('Day of Week')
            axes[1, 1].set_ylabel('Average Delay (minutes)')
            axes[1, 1].set_title('Average Delay by Day of Week')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def visualize_route_analysis(self, save_path='visualizations/route_analysis.png'):
        """Analyze and visualize route performance"""
        if 'origin' not in self.df.columns or 'destination' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Route Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top routes by frequency
        routes = self.df.groupby(['origin', 'destination']).size().reset_index(name='count')
        routes = routes.sort_values('count', ascending=False).head(10)
        route_labels = [f"{row['origin']} → {row['destination']}" for _, row in routes.iterrows()]
        axes[0, 0].barh(route_labels, routes['count'].values, color='coral')
        axes[0, 0].set_xlabel('Number of Flights')
        axes[0, 0].set_title('Top 10 Busiest Routes')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # 2. Route delay analysis
        route_delays = self.df.groupby(['origin', 'destination'])['departure_delay'].mean().reset_index()
        route_delays = route_delays.sort_values('departure_delay', ascending=False).head(10)
        route_delay_labels = [f"{row['origin']} → {row['destination']}" for _, row in route_delays.iterrows()]
        axes[0, 1].barh(route_delay_labels, route_delays['departure_delay'].values, color='indianred')
        axes[0, 1].set_xlabel('Average Delay (minutes)')
        axes[0, 1].set_title('Top 10 Routes with Highest Delays')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. Airport traffic (departures)
        if 'origin' in self.df.columns:
            airport_departures = self.df['origin'].value_counts().head(10)
            axes[1, 0].bar(range(len(airport_departures)), airport_departures.values, color='skyblue')
            axes[1, 0].set_xticks(range(len(airport_departures)))
            axes[1, 0].set_xticklabels(airport_departures.index, rotation=45, ha='right')
            axes[1, 0].set_ylabel('Number of Departures')
            axes[1, 0].set_title('Top 10 Busiest Airports (Departures)')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Distance vs Delay scatter
        if 'distance' in self.df.columns and 'departure_delay' in self.df.columns:
            sample_df = self.df.sample(min(1000, len(self.df)))  # Sample for performance
            axes[1, 1].scatter(sample_df['distance'], sample_df['departure_delay'], 
                             alpha=0.5, s=20, color='purple')
            axes[1, 1].set_xlabel('Flight Distance (km)')
            axes[1, 1].set_ylabel('Departure Delay (minutes)')
            axes[1, 1].set_title('Flight Distance vs Delay')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def create_interactive_dashboard(self, save_path='visualizations/interactive_dashboard.html'):
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delay Distribution by Airline', 
                          'Delay Trends Over Time',
                          'Route Performance Heatmap',
                          'On-Time Performance by Hour'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # 1. Delay by airline
        if 'airline' in self.df.columns:
            airline_delays = self.df.groupby('airline')['departure_delay'].mean().sort_values()
            fig.add_trace(
                go.Bar(x=airline_delays.index, y=airline_delays.values, 
                      name='Avg Delay', marker_color='steelblue'),
                row=1, col=1
            )
        
        # 2. Delay trends over time
        if 'date' in self.df.columns:
            daily_delays = self.df.groupby(self.df['date'].dt.date)['departure_delay'].mean()
            fig.add_trace(
                go.Scatter(x=daily_delays.index, y=daily_delays.values,
                          mode='lines+markers', name='Daily Avg Delay',
                          line=dict(color='coral', width=2)),
                row=1, col=2
            )
        
        # 3. Heatmap: Delay by day of week and hour
        if 'day_of_week' in self.df.columns and 'hour' in self.df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot_data = self.df.pivot_table(
                values='departure_delay', 
                index='day_of_week', 
                columns='hour', 
                aggfunc='mean'
            )
            pivot_data = pivot_data.reindex([d for d in day_order if d in pivot_data.index])
            
            fig.add_trace(
                go.Heatmap(z=pivot_data.values, x=pivot_data.columns, y=pivot_data.index,
                          colorscale='RdYlGn_r', name='Delay Heatmap'),
                row=2, col=1
            )
        
        # 4. On-time performance by hour
        if 'hour' in self.df.columns:
            hourly_ontime = self.df.groupby('hour').apply(
                lambda x: (x['departure_delay'] <= 0).sum() / len(x) * 100
            )
            fig.add_trace(
                go.Bar(x=hourly_ontime.index, y=hourly_ontime.values,
                      name='On-Time %', marker_color='green'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Aviation Analytics Interactive Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Airline", row=1, col=1)
        fig.update_yaxes(title_text="Avg Delay (min)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Avg Delay (min)", row=1, col=2)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="Day of Week", row=2, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
        fig.update_yaxes(title_text="On-Time %", row=2, col=2)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"✓ Saved interactive dashboard: {save_path}")
    
    def generate_report(self, save_path='aviation_analytics_report.txt'):
        """Generate text report with key insights"""
        stats = self.overview_statistics()
        
        report = f"""
{'='*60}
AVIATION DATA ANALYTICS REPORT
{'='*60}

OVERVIEW STATISTICS
{'-'*60}
Total Flights Analyzed: {stats['total_flights']:,}
Number of Airlines: {stats['total_airlines']}
Number of Airports: {stats['total_airports']}
Average Departure Delay: {stats['avg_departure_delay']:.2f} minutes
On-Time Performance: {stats['on_time_percentage']:.2f}%
Average Flight Duration: {stats['avg_flight_duration']:.2f} hours

KEY INSIGHTS
{'-'*60}
"""
        
        # Add airline analysis
        if 'airline' in self.df.columns:
            airline_perf = self.df.groupby('airline').agg({
                'departure_delay': 'mean',
                'flight_duration': 'mean'
            }).round(2)
            report += "\nAIRLINE PERFORMANCE:\n"
            for airline, row in airline_perf.iterrows():
                report += f"  {airline}: Avg Delay = {row['departure_delay']:.1f} min, "
                report += f"Avg Duration = {row['flight_duration']:.1f} hrs\n"
        
        # Add delay category breakdown
        if 'delay_category' in self.df.columns:
            delay_breakdown = self.df['delay_category'].value_counts(normalize=True) * 100
            report += "\nDELAY CATEGORY BREAKDOWN:\n"
            for category, pct in delay_breakdown.items():
                report += f"  {category}: {pct:.1f}%\n"
        
        # Add recommendations
        report += f"""
RECOMMENDATIONS
{'-'*60}
1. Focus on routes with consistently high delays for operational improvements
2. Analyze peak delay times to optimize scheduling
3. Monitor airline-specific performance metrics
4. Consider weather and seasonal factors in delay patterns
5. Implement predictive analytics for delay forecasting

{'='*60}
Report generated using Python data analytics tools
Perfect for showcasing skills in aviation data analysis
{'='*60}
"""
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Saved report: {save_path}")
        print(report)
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*60)
        print("AVIATION DATA ANALYTICS - FULL ANALYSIS")
        print("="*60 + "\n")
        
        if self.df is None:
            print("✗ No data loaded. Cannot proceed.")
            return
        
        print("Generating visualizations...")
        self.visualize_delay_distribution()
        self.visualize_route_analysis()
        self.create_interactive_dashboard()
        self.generate_report()
        
        print("\n" + "="*60)
        print("✓ Analysis complete! Check the 'visualizations' folder")
        print("="*60 + "\n")

if __name__ == "__main__":
    # Initialize analytics
    analytics = FlightAnalytics()
    
    # Run full analysis
    analytics.run_full_analysis()
    
    # Run advanced analytics if available
    try:
        from advanced_analytics import AdvancedFlightAnalytics
        print("\n" + "="*60)
        print("RUNNING ADVANCED ANALYTICS")
        print("="*60)
        advanced = AdvancedFlightAnalytics(analytics.df)
        advanced.run_advanced_analysis()
    except ImportError:
        print("\nNote: Install scikit-learn for advanced analytics features")
        print("  pip install scikit-learn")

