import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_web_analytics_data(url, start_date, end_date, seed=42):
    """
    Generate realistic web analytics data for a single URL by day
    
    Parameters:
    - url: The URL to generate data for
    - start_date: Start date (YYYY-MM-DD format)
    - end_date: End date (YYYY-MM-DD format)
    - seed: Random seed for reproducible results
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Convert dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate date range
    date_range = []
    current_date = start
    while current_date <= end:
        date_range.append(current_date)
        current_date += timedelta(days=1)
    
    # Marketing segments with realistic distributions
    marketing_segments = [
        'Organic Search', 'Paid Search', 'Social Media', 'Direct Traffic',
        'Email Marketing', 'Referral', 'Display Ads', 'Video Ads'
    ]
    
    # Base weights for marketing segments (realistic distribution)
    segment_weights = [0.35, 0.20, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02]
    
    data = []
    
    # Track cumulative unique visitors for return visitor calculation
    cumulative_unique_visitors = set()
    
    for date in date_range:
        # Add day-of-week and seasonal effects
        day_of_week = date.weekday()  # 0=Monday, 6=Sunday
        is_weekend = day_of_week >= 5
        
        # Weekend traffic is typically lower
        weekend_multiplier = 0.7 if is_weekend else 1.0
        
        # Seasonal effect (simple sine wave)
        seasonal_effect = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        
        # Base page visits (with some randomness)
        base_visits = int(np.random.normal(1000, 200) * weekend_multiplier * seasonal_effect)
        base_visits = max(base_visits, 50)  # Minimum visits
        
        # Generate data for each marketing segment
        segment_data = []
        total_visits = 0
        total_unique = 0
        
        for segment, weight in zip(marketing_segments, segment_weights):
            # Calculate visits for this segment
            segment_visits = int(base_visits * weight * np.random.uniform(0.8, 1.2))
            segment_visits = max(segment_visits, 1)
            
            # Unique visitors (typically 60-80% of page visits)
            unique_ratio = np.random.uniform(0.6, 0.8)
            segment_unique = int(segment_visits * unique_ratio)
            segment_unique = max(segment_unique, 1)
            
            # Return visitors (based on historical unique visitors)
            if len(cumulative_unique_visitors) > 0:
                # Return visitor rate varies by segment
                return_rates = {
                    'Direct Traffic': 0.4, 'Email Marketing': 0.35, 'Organic Search': 0.25,
                    'Social Media': 0.2, 'Referral': 0.15, 'Paid Search': 0.1,
                    'Display Ads': 0.08, 'Video Ads': 0.05
                }
                base_return_rate = return_rates.get(segment, 0.2)
                segment_return = int(segment_unique * base_return_rate * np.random.uniform(0.5, 1.5))
                segment_return = min(segment_return, min(segment_unique, len(cumulative_unique_visitors)))
            else:
                segment_return = 0
            
            segment_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'url': url,
                'marketing_segment': segment,
                'page_visits': segment_visits,
                'unique_visitors': segment_unique,
                'return_visitors': segment_return
            })
            
            total_visits += segment_visits
            total_unique += segment_unique
            
            # Add new unique visitors to cumulative set
            new_unique = segment_unique - segment_return
            for i in range(new_unique):
                cumulative_unique_visitors.add(f"{date}_{segment}_{i}")
        
        data.extend(segment_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date and marketing segment
    df = df.sort_values(['date', 'marketing_segment']).reset_index(drop=True)
    
    return df

# Example usage
if __name__ == "__main__":
    # Generate data for a sample URL
    url = "https://example.com/product-page"
    start_date = "2024-01-01"
    end_date = "2025-07-31"
    
    # Generate the data
    analytics_data = generate_web_analytics_data(url, start_date, end_date)
    
    # Display sample data
    print("Sample Web Analytics Data:")
    print("=" * 50)
    print(analytics_data.head(20))
    
    print(f"\nTotal records generated: {len(analytics_data)}")
    print(f"Date range: {analytics_data['date'].min()} to {analytics_data['date'].max()}")
    print(f"Marketing segments: {analytics_data['marketing_segment'].unique()}")
    
    # Summary statistics
    print("\nDaily Totals Summary:")
    print("=" * 30)
    daily_summary = analytics_data.groupby('date').agg({
        'page_visits': 'sum',
        'unique_visitors': 'sum',
        'return_visitors': 'sum'
    }).round(2)
    print(daily_summary.head(10))
    
    # Marketing segment summary
    print("\nMarketing Segment Summary:")
    print("=" * 35)
    segment_summary = analytics_data.groupby('marketing_segment').agg({
        'page_visits': ['sum', 'mean'],
        'unique_visitors': ['sum', 'mean'],
        'return_visitors': ['sum', 'mean']
    }).round(2)
    print(segment_summary)
    
    # Save to CSV
    filename = f"web_analytics_data_{start_date}_to_{end_date}.csv"
    analytics_data.to_csv(filename, index=False)
    print(f"\nData saved to: {filename}")
    
    # Optional: Create a more detailed dataset with additional metrics
    print("\n" + "="*60)
    print("GENERATING EXTENDED DATASET WITH ADDITIONAL METRICS")
    print("="*60)
    
    def generate_extended_analytics_data(url, start_date, end_date, seed=42):
        """Extended version with additional realistic metrics"""
        base_data = generate_web_analytics_data(url, start_date, end_date, seed)
        
        # Add additional realistic metrics
        extended_data = base_data.copy()
        
        # Add bounce rate (varies by marketing segment)
        bounce_rates = {
            'Direct Traffic': 0.3, 'Email Marketing': 0.25, 'Organic Search': 0.4,
            'Social Media': 0.6, 'Referral': 0.45, 'Paid Search': 0.35,
            'Display Ads': 0.7, 'Video Ads': 0.5
        }
        
        extended_data['bounce_rate'] = extended_data['marketing_segment'].map(bounce_rates)
        extended_data['bounce_rate'] *= np.random.uniform(0.8, 1.2, len(extended_data))
        extended_data['bounce_rate'] = extended_data['bounce_rate'].clip(0, 1).round(3)
        
        # Add average session duration (in minutes)
        session_durations = {
            'Direct Traffic': 4.5, 'Email Marketing': 5.2, 'Organic Search': 3.8,
            'Social Media': 2.1, 'Referral': 3.2, 'Paid Search': 2.8,
            'Display Ads': 1.5, 'Video Ads': 6.2
        }
        
        extended_data['avg_session_duration'] = extended_data['marketing_segment'].map(session_durations)
        extended_data['avg_session_duration'] *= np.random.uniform(0.7, 1.3, len(extended_data))
        extended_data['avg_session_duration'] = extended_data['avg_session_duration'].round(2)
        
        # Add conversion rate (varies by segment)
        conversion_rates = {
            'Direct Traffic': 0.035, 'Email Marketing': 0.045, 'Organic Search': 0.025,
            'Social Media': 0.015, 'Referral': 0.020, 'Paid Search': 0.040,
            'Display Ads': 0.008, 'Video Ads': 0.012
        }
        
        extended_data['conversion_rate'] = extended_data['marketing_segment'].map(conversion_rates)
        extended_data['conversion_rate'] *= np.random.uniform(0.5, 1.5, len(extended_data))
        extended_data['conversion_rate'] = extended_data['conversion_rate'].clip(0, 0.1).round(4)
        
        # Calculate conversions
        extended_data['conversions'] = (extended_data['unique_visitors'] * extended_data['conversion_rate']).round().astype(int)
        
        return extended_data
    
    # Generate extended dataset
    extended_data = generate_extended_analytics_data(url, start_date, end_date)
    
    print("Extended Analytics Data Sample:")
    print(extended_data.head(10))
    
    # Save extended data
    extended_filename = f"extended_web_analytics_{start_date}_to_{end_date}.csv"
    extended_data.to_csv(extended_filename, index=False)
    print(f"\nExtended data saved to: {extended_filename}")