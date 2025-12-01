"""
Advanced Analytics Module for Aviation Data
Includes predictive analytics, correlation analysis, and advanced visualizations
Perfect for showcasing advanced data analytics skills
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedFlightAnalytics:
    """Advanced analytics including predictive modeling and correlation analysis"""
    
    def __init__(self, df):
        """Initialize with flight dataframe"""
        self.df = df.copy()
        self.model = None
        self.feature_importance = None
    
    def correlation_analysis(self, save_path='visualizations/correlation_analysis.png'):
        """Analyze correlations between flight variables"""
        # Select numeric columns
        numeric_cols = ['departure_delay', 'arrival_delay', 'flight_duration', 
                       'distance', 'hour', 'month']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            print("Not enough numeric columns for correlation analysis")
            return
        
        corr_data = self.df[available_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix: Flight Performance Variables', 
                    fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
        
        return corr_data
    
    def delay_prediction_model(self):
        """Build a machine learning model to predict flight delays"""
        print("\nBuilding delay prediction model...")
        
        # Prepare features
        feature_cols = ['hour', 'month', 'distance', 'flight_duration']
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        if 'day_of_week' in self.df.columns:
            # Encode day of week
            day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 
                          'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
            self.df['day_encoded'] = self.df['day_of_week'].map(day_mapping)
            available_features.append('day_encoded')
        
        if len(available_features) < 2:
            print("Not enough features for modeling")
            return None
        
        # Prepare data
        X = self.df[available_features].fillna(0)
        y = self.df['departure_delay'].fillna(0)
        
        # Remove outliers
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            print("Not enough data after outlier removal")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  Model Performance:")
        print(f"    Mean Absolute Error: {mae:.2f} minutes")
        print(f"    R² Score: {r2:.3f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': self.model,
            'mae': mae,
            'r2': r2,
            'feature_importance': self.feature_importance
        }
    
    def visualize_feature_importance(self, save_path='visualizations/feature_importance.png'):
        """Visualize feature importance from the prediction model"""
        if self.feature_importance is None:
            print("Please run delay_prediction_model() first")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.feature_importance)))
        bars = ax.barh(self.feature_importance['feature'], 
                      self.feature_importance['importance'], 
                      color=colors)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Feature Importance: Delay Prediction Model', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(self.feature_importance.iterrows()):
            ax.text(row['importance'] + 0.01, i, f"{row['importance']:.3f}", 
                   va='center', fontsize=10)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def statistical_summary(self):
        """Generate comprehensive statistical summary"""
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        
        numeric_cols = ['departure_delay', 'arrival_delay', 'flight_duration', 'distance']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if not available_cols:
            print("No numeric columns available for statistical analysis")
            return
        
        stats_df = self.df[available_cols].describe()
        print("\nDescriptive Statistics:")
        print(stats_df)
        
        # Additional statistics
        print("\n" + "-"*60)
        print("Additional Insights:")
        print("-"*60)
        
        if 'departure_delay' in self.df.columns:
            delays = self.df['departure_delay']
            print(f"\nDelay Analysis:")
            print(f"  Median delay: {delays.median():.2f} minutes")
            print(f"  Standard deviation: {delays.std():.2f} minutes")
            print(f"  Skewness: {stats.skew(delays):.2f}")
            print(f"  Kurtosis: {stats.kurtosis(delays):.2f}")
            
            # Percentiles
            print(f"\nDelay Percentiles:")
            for p in [25, 50, 75, 90, 95, 99]:
                print(f"  {p}th percentile: {np.percentile(delays, p):.2f} minutes")
        
        return stats_df
    
    def time_series_analysis(self, save_path='visualizations/time_series_analysis.png'):
        """Analyze delay trends over time"""
        if 'date' not in self.df.columns:
            print("Date column not available for time series analysis")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Time Series Analysis: Flight Delays', fontsize=16, fontweight='bold')
        
        # Daily average delays
        daily_delays = self.df.groupby(self.df['date'].dt.date)['departure_delay'].mean()
        axes[0].plot(daily_delays.index, daily_delays.values, linewidth=2, color='steelblue')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_title('Daily Average Departure Delays', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Average Delay (minutes)')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Weekly average delays
        self.df['week'] = self.df['date'].dt.isocalendar().week
        weekly_delays = self.df.groupby('week')['departure_delay'].mean()
        axes[1].bar(weekly_delays.index, weekly_delays.values, color='coral', alpha=0.7)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_title('Weekly Average Departure Delays', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Week Number')
        axes[1].set_ylabel('Average Delay (minutes)')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Monthly average delays
        monthly_delays = self.df.groupby('month')['departure_delay'].mean()
        axes[2].bar(monthly_delays.index, monthly_delays.values, color='purple', alpha=0.7)
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[2].set_title('Monthly Average Departure Delays', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Month')
        axes[2].set_ylabel('Average Delay (minutes)')
        axes[2].set_xticks(range(1, 13))
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def create_comprehensive_dashboard(self, save_path='visualizations/comprehensive_dashboard.html'):
        """Create a comprehensive interactive dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Delay Distribution by Airline',
                'Delay Trends Over Time',
                'Correlation Heatmap',
                'On-Time Performance by Hour',
                'Route Performance (Top 15)',
                'Delay Prediction vs Actual'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Delay by airline
        if 'airline' in self.df.columns:
            airline_delays = self.df.groupby('airline')['departure_delay'].mean().sort_values()
            fig.add_trace(
                go.Bar(x=airline_delays.index, y=airline_delays.values,
                      name='Avg Delay', marker_color='steelblue', showlegend=False),
                row=1, col=1
            )
        
        # 2. Delay trends over time
        if 'date' in self.df.columns:
            daily_delays = self.df.groupby(self.df['date'].dt.date)['departure_delay'].mean()
            fig.add_trace(
                go.Scatter(x=list(daily_delays.index), y=daily_delays.values,
                          mode='lines+markers', name='Daily Avg Delay',
                          line=dict(color='coral', width=2), showlegend=False),
                row=1, col=2
            )
        
        # 3. Correlation heatmap
        numeric_cols = ['departure_delay', 'arrival_delay', 'flight_duration', 'distance', 'hour']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        if len(available_cols) >= 2:
            corr_data = self.df[available_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_data.values, x=corr_data.columns, y=corr_data.index,
                          colorscale='RdBu', zmid=0, showscale=True, showlegend=False),
                row=2, col=1
            )
        
        # 4. On-time performance by hour
        if 'hour' in self.df.columns:
            hourly_ontime = self.df.groupby('hour').apply(
                lambda x: (x['departure_delay'] <= 0).sum() / len(x) * 100
            )
            fig.add_trace(
                go.Bar(x=hourly_ontime.index, y=hourly_ontime.values,
                      name='On-Time %', marker_color='green', showlegend=False),
                row=2, col=2
            )
        
        # 5. Top routes by frequency
        if 'origin' in self.df.columns and 'destination' in self.df.columns:
            routes = self.df.groupby(['origin', 'destination']).size().reset_index(name='count')
            routes = routes.sort_values('count', ascending=False).head(15)
            route_labels = [f"{row['origin']}→{row['destination']}" for _, row in routes.iterrows()]
            fig.add_trace(
                go.Bar(x=route_labels, y=routes['count'].values,
                      marker_color='purple', showlegend=False),
                row=3, col=1
            )
        
        # 6. Delay distribution
        fig.add_trace(
            go.Histogram(x=self.df['departure_delay'], nbinsx=50,
                        marker_color='orange', showlegend=False),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Aviation Analytics Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Airline", row=1, col=1)
        fig.update_yaxes(title_text="Avg Delay (min)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Avg Delay (min)", row=1, col=2)
        fig.update_xaxes(title_text="Variable", row=2, col=1)
        fig.update_yaxes(title_text="Variable", row=2, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
        fig.update_yaxes(title_text="On-Time %", row=2, col=2)
        fig.update_xaxes(title_text="Route", row=3, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Flight Count", row=3, col=1)
        fig.update_xaxes(title_text="Delay (minutes)", row=3, col=2)
        fig.update_yaxes(title_text="Frequency", row=3, col=2)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"✓ Saved comprehensive dashboard: {save_path}")
    
    def run_advanced_analysis(self):
        """Run all advanced analytics"""
        print("\n" + "="*60)
        print("ADVANCED ANALYTICS - FULL ANALYSIS")
        print("="*60 + "\n")
        
        # Correlation analysis
        print("1. Running correlation analysis...")
        self.correlation_analysis()
        
        # Statistical summary
        print("\n2. Generating statistical summary...")
        self.statistical_summary()
        
        # Time series analysis
        print("\n3. Performing time series analysis...")
        self.time_series_analysis()
        
        # Predictive modeling
        print("\n4. Building delay prediction model...")
        model_results = self.delay_prediction_model()
        if model_results:
            self.visualize_feature_importance()
        
        # Comprehensive dashboard
        print("\n5. Creating comprehensive dashboard...")
        self.create_comprehensive_dashboard()
        
        print("\n" + "="*60)
        print("✓ Advanced analysis complete!")
        print("="*60 + "\n")

