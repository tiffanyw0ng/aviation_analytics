"""
Interactive Streamlit Dashboard for Aviation Analytics
Run with: streamlit run dashboard.py
"""

import streamlit as st
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

# Page configuration
st.set_page_config(
    page_title="Aviation Analytics Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load flight data"""
    data_path = Path('data/flight_delays_sample.csv')
    if not data_path.exists():
        st.error("Data file not found! Please run data_downloader.py first.")
        return None
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def main():
    # Header
    st.markdown('<div class="main-header">✈️ Aviation Data Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Cathay Pacific Digital & IT Summer Internship Programme 2026</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Airline filter
    airlines = ['All'] + sorted(df['airline'].unique().tolist())
    selected_airline = st.sidebar.selectbox("Select Airline", airlines)
    
    # Date range filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Delay category filter
    delay_categories = ['All'] + df['delay_category'].unique().tolist()
    selected_category = st.sidebar.selectbox("Delay Category", delay_categories)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_airline != 'All':
        filtered_df = filtered_df[filtered_df['airline'] == selected_airline]
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) &
            (filtered_df['date'].dt.date <= date_range[1])
        ]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['delay_category'] == selected_category]
    
    # Key Metrics
    st.markdown('<div class="sub-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_flights = len(filtered_df)
        st.metric("Total Flights", f"{total_flights:,}")
    
    with col2:
        avg_delay = filtered_df['departure_delay'].mean()
        st.metric("Avg Delay", f"{avg_delay:.1f} min", 
                 delta=f"{avg_delay - df['departure_delay'].mean():.1f} vs Overall")
    
    with col3:
        on_time_pct = (filtered_df['departure_delay'] <= 0).sum() / len(filtered_df) * 100
        st.metric("On-Time %", f"{on_time_pct:.1f}%")
    
    with col4:
        total_routes = filtered_df.groupby(['origin', 'destination']).ngroups
        st.metric("Unique Routes", f"{total_routes}")
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Delay Analysis", 
        "🛫 Route Performance", 
        "⏰ Temporal Trends",
        "🔗 Correlations",
        "💡 Insights"
    ])
    
    # Tab 1: Delay Analysis
    with tab1:
        st.markdown('<div class="sub-header">Flight Delay Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Delay distribution
            fig = px.histogram(
                filtered_df, 
                x='departure_delay',
                nbins=50,
                title='Delay Distribution',
                labels={'departure_delay': 'Departure Delay (minutes)', 'count': 'Frequency'},
                color_discrete_sequence=['steelblue']
            )
            fig.add_vline(x=filtered_df['departure_delay'].mean(), 
                         line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {filtered_df['departure_delay'].mean():.1f} min")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Delay by airline
            airline_delays = filtered_df.groupby('airline')['departure_delay'].mean().sort_values()
            fig = px.bar(
                x=airline_delays.values,
                y=airline_delays.index,
                orientation='h',
                title='Average Delay by Airline',
                labels={'x': 'Average Delay (minutes)', 'y': 'Airline'},
                color=airline_delays.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Delay category breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            delay_counts = filtered_df['delay_category'].value_counts()
            fig = px.pie(
                values=delay_counts.values,
                names=delay_counts.index,
                title='Delay Category Distribution',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Delay by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_delays = filtered_df.groupby('day_of_week')['departure_delay'].mean()
            day_delays = day_delays.reindex([d for d in day_order if d in day_delays.index])
            fig = px.line(
                x=day_delays.index,
                y=day_delays.values,
                markers=True,
                title='Average Delay by Day of Week',
                labels={'x': 'Day of Week', 'y': 'Average Delay (minutes)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Route Performance
    with tab2:
        st.markdown('<div class="sub-header">Route Performance Analysis</div>', unsafe_allow_html=True)
        
        # Top routes
        routes = filtered_df.groupby(['origin', 'destination']).agg({
            'departure_delay': 'mean',
            'flight_duration': 'mean',
            'distance': 'mean'
        }).reset_index()
        routes['route'] = routes['origin'] + ' → ' + routes['destination']
        routes['flight_count'] = filtered_df.groupby(['origin', 'destination']).size().values
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 busiest routes
            top_routes = routes.nlargest(10, 'flight_count')
            fig = px.bar(
                x=top_routes['flight_count'],
                y=top_routes['route'],
                orientation='h',
                title='Top 10 Busiest Routes',
                labels={'x': 'Number of Flights', 'y': 'Route'},
                color=top_routes['flight_count'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Routes with highest delays
            high_delay_routes = routes.nlargest(10, 'departure_delay')
            fig = px.bar(
                x=high_delay_routes['departure_delay'],
                y=high_delay_routes['route'],
                orientation='h',
                title='Top 10 Routes with Highest Delays',
                labels={'x': 'Average Delay (minutes)', 'y': 'Route'},
                color=high_delay_routes['departure_delay'],
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distance vs Delay scatter
        fig = px.scatter(
            filtered_df.sample(min(2000, len(filtered_df))),
            x='distance',
            y='departure_delay',
            color='airline',
            size='flight_duration',
            hover_data=['origin', 'destination', 'day_of_week'],
            title='Flight Distance vs Departure Delay',
            labels={'distance': 'Distance (km)', 'departure_delay': 'Departure Delay (minutes)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Temporal Trends
    with tab3:
        st.markdown('<div class="sub-header">Temporal Analysis</div>', unsafe_allow_html=True)
        
        # Daily trends
        daily_delays = filtered_df.groupby(filtered_df['date'].dt.date)['departure_delay'].mean()
        fig = px.line(
            x=daily_delays.index,
            y=daily_delays.values,
            title='Daily Average Departure Delays',
            labels={'x': 'Date', 'y': 'Average Delay (minutes)'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Delay by hour
            hourly_delays = filtered_df.groupby('hour')['departure_delay'].mean()
            fig = px.bar(
                x=hourly_delays.index,
                y=hourly_delays.values,
                title='Average Delay by Hour of Day',
                labels={'x': 'Hour of Day', 'y': 'Average Delay (minutes)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Delay by month
            monthly_delays = filtered_df.groupby('month')['departure_delay'].mean()
            fig = px.bar(
                x=monthly_delays.index,
                y=monthly_delays.values,
                title='Average Delay by Month',
                labels={'x': 'Month', 'y': 'Average Delay (minutes)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap: Day of week vs Hour
        pivot_data = filtered_df.pivot_table(
            values='departure_delay',
            index='day_of_week',
            columns='hour',
            aggfunc='mean'
        )
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pivot_data.reindex([d for d in day_order if d in pivot_data.index])
        
        fig = px.imshow(
            pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale='RdYlGn_r',
            title='Delay Heatmap: Day of Week vs Hour of Day',
            labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Avg Delay (min)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Correlations
    with tab4:
        st.markdown('<div class="sub-header">Correlation Analysis</div>', unsafe_allow_html=True)
        
        numeric_cols = ['departure_delay', 'arrival_delay', 'flight_duration', 'distance', 'hour', 'month']
        available_cols = [col for col in numeric_cols if col in filtered_df.columns]
        
        if len(available_cols) >= 2:
            corr_matrix = filtered_df[available_cols].corr()
            
            fig = px.imshow(
                corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                title='Correlation Matrix: Flight Performance Variables',
                labels={'color': 'Correlation'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display correlation values
            st.subheader("Correlation Values")
            st.dataframe(corr_matrix.round(3), use_container_width=True)
    
    # Tab 5: Insights
    with tab5:
        st.markdown('<div class="sub-header">Key Business Insights</div>', unsafe_allow_html=True)
        
        # Best/worst performing airlines
        airline_perf = filtered_df.groupby('airline')['departure_delay'].mean().sort_values()
        best_airline = airline_perf.idxmin()
        worst_airline = airline_perf.idxmax()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🏆 **Best Performing Airline**: {best_airline}")
            st.write(f"Average Delay: {airline_perf[best_airline]:.1f} minutes")
        
        with col2:
            st.warning(f"⚠️ **Airline Needing Improvement**: {worst_airline}")
            st.write(f"Average Delay: {airline_perf[worst_airline]:.1f} minutes")
        
        # Best/worst days
        day_perf = filtered_df.groupby('day_of_week')['departure_delay'].mean().sort_values()
        best_day = day_perf.idxmin()
        worst_day = day_perf.idxmax()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"📅 **Best Day to Fly**: {best_day}")
            st.write(f"Average Delay: {day_perf[best_day]:.1f} minutes")
        
        with col2:
            st.error(f"📅 **Worst Day to Fly**: {worst_day}")
            st.write(f"Average Delay: {day_perf[worst_day]:.1f} minutes")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        recommendations = [
            "Focus operational improvements on routes with consistently high delays",
            "Optimize scheduling during peak delay periods (check temporal trends)",
            "Implement predictive analytics for delay forecasting",
            "Analyze root causes of delays by airline and route",
            "Consider weather and seasonal factors in delay patterns",
            "Monitor and benchmark performance against industry standards"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Download report
        st.markdown("---")
        st.subheader("📥 Download Report")
        
        report_text = f"""
AVIATION DATA ANALYTICS REPORT
============================================================

OVERVIEW STATISTICS
------------------------------------------------------------
Total Flights Analyzed: {len(filtered_df):,}
Number of Airlines: {filtered_df['airline'].nunique()}
Number of Airports: {pd.concat([filtered_df['origin'], filtered_df['destination']]).nunique()}
Average Departure Delay: {filtered_df['departure_delay'].mean():.2f} minutes
On-Time Performance: {(filtered_df['departure_delay'] <= 0).sum() / len(filtered_df) * 100:.2f}%

KEY INSIGHTS
------------------------------------------------------------
Best Performing Airline: {best_airline} ({airline_perf[best_airline]:.1f} min avg delay)
Worst Performing Airline: {worst_airline} ({airline_perf[worst_airline]:.1f} min avg delay)
Best Day to Fly: {best_day} ({day_perf[best_day]:.1f} min avg delay)
Worst Day to Fly: {worst_day} ({day_perf[worst_day]:.1f} min avg delay)
"""
        
        st.download_button(
            label="Download Text Report",
            data=report_text,
            file_name="aviation_analytics_report.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()

