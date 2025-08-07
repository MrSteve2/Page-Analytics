import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WebAnalyticsAnalyzer:
    def __init__(self, filename):
        """
        Initialize the analyzer with web analytics data
        
        Parameters:
        filename (str): Path to the CSV file containing web analytics data
        """
        self.filename = filename
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and prepare the web analytics data"""
        try:
            self.df = pd.read_csv(self.filename)
            print(f"Data loaded successfully: {len(self.df)} records")
            
            # Validate expected columns
            expected_columns = ['date', 'url', 'marketing_segment', 'page_visits', 'unique_visitors', 'return_visitors']
            missing_columns = [col for col in expected_columns if col not in self.df.columns]
            
            if missing_columns:
                print(f"Warning: Missing expected columns: {missing_columns}")
                print(f"Available columns: {list(self.df.columns)}")
            
            # Convert date column to datetime
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Display basic info about the dataset
            print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
            print(f"Number of unique URLs: {self.df['url'].nunique() if 'url' in self.df.columns else 'N/A'}")
            print(f"Marketing segments: {self.df['marketing_segment'].unique() if 'marketing_segment' in self.df.columns else 'N/A'}")
            print(f"Columns: {list(self.df.columns)}")
            print("\nFirst few rows:")
            print(self.df.head())
            
        except FileNotFoundError:
            print(f"Error: File {self.filename} not found.")
            print("Please ensure the file exists in the current directory.")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def basic_statistics(self):
        """Generate basic statistical summary"""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print("BASIC STATISTICS SUMMARY")
        print("="*60)
        
        # Numeric columns summary
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\nNumeric Columns Summary:")
            print(self.df[numeric_cols].describe().round(2))
        
        # Categorical columns summary
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'date']
        
        if len(categorical_cols) > 0:
            print(f"\nCategorical Columns Summary:")
            for col in categorical_cols:
                print(f"\n{col}:")
                print(self.df[col].value_counts())
    
    def daily_analysis(self):
        """Analyze daily trends and patterns"""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print("DAILY ANALYSIS")
        print("="*60)
        
        # Group by date for daily totals
        daily_metrics = self.df.groupby('date').agg({
            'page_visits': 'sum',
            'unique_visitors': 'sum', 
            'return_visitors': 'sum'
        }).reset_index()
        
        # Add calculated metrics
        daily_metrics['new_visitors'] = daily_metrics['unique_visitors'] - daily_metrics['return_visitors']
        daily_metrics['return_visitor_rate'] = (daily_metrics['return_visitors'] / daily_metrics['unique_visitors'] * 100).round(2)
        daily_metrics['pages_per_visitor'] = (daily_metrics['page_visits'] / daily_metrics['unique_visitors']).round(2)
        
        # Add day of week analysis
        daily_metrics['day_of_week'] = daily_metrics['date'].dt.day_name()
        daily_metrics['is_weekend'] = daily_metrics['date'].dt.weekday >= 5
        
        print("Daily Metrics Summary:")
        print(daily_metrics.describe().round(2))
        
        print(f"\nTop 10 Days by Page Visits:")
        top_days = daily_metrics.nlargest(10, 'page_visits')[['date', 'page_visits', 'unique_visitors', 'return_visitors']]
        print(top_days)
        
        # Day of week analysis
        print(f"\nDay of Week Analysis:")
        dow_analysis = daily_metrics.groupby('day_of_week').agg({
            'page_visits': ['mean', 'sum'],
            'unique_visitors': ['mean', 'sum'],
            'return_visitors': ['mean', 'sum']
        }).round(2)
        print(dow_analysis)
        
        # Weekend vs Weekday comparison
        print(f"\nWeekend vs Weekday Comparison:")
        weekend_comparison = daily_metrics.groupby('is_weekend').agg({
            'page_visits': ['mean', 'sum'],
            'unique_visitors': ['mean', 'sum'],
            'return_visitor_rate': 'mean',
            'pages_per_visitor': 'mean'
        }).round(2)
        weekend_comparison.index = ['Weekday', 'Weekend']
        print(weekend_comparison)
        
        return daily_metrics
    
    def url_analysis(self):
        """Analyze performance by URL"""
        if self.df is None or 'url' not in self.df.columns:
            print("URL data not available")
            return None
        
        print("\n" + "="*60)
        print("URL PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Aggregate by URL
        url_metrics = self.df.groupby('url').agg({
            'page_visits': ['sum', 'mean', 'std'],
            'unique_visitors': ['sum', 'mean', 'std'],
            'return_visitors': ['sum', 'mean', 'std']
        }).round(2)
        
        # Calculate additional URL metrics
        url_summary = self.df.groupby('url').agg({
            'page_visits': 'sum',
            'unique_visitors': 'sum',
            'return_visitors': 'sum'
        })
        
        url_summary['return_visitor_rate'] = (url_summary['return_visitors'] / url_summary['unique_visitors'] * 100).round(2)
        url_summary['pages_per_visitor'] = (url_summary['page_visits'] / url_summary['unique_visitors']).round(2)
        url_summary['traffic_share'] = (url_summary['page_visits'] / url_summary['page_visits'].sum() * 100).round(2)
        url_summary['new_visitors'] = url_summary['unique_visitors'] - url_summary['return_visitors']
        
        print("URL Performance Summary:")
        print(url_summary.sort_values('page_visits', ascending=False))
        
        print(f"\nNumber of unique URLs analyzed: {len(url_summary)}")
        
        # Top performing URLs
        print(f"\nTop 5 URLs by Page Visits:")
        top_urls = url_summary.nlargest(5, 'page_visits')[['page_visits', 'unique_visitors', 'return_visitor_rate']]
        print(top_urls)
        
        # URLs with highest return visitor rate
        print(f"\nTop 5 URLs by Return Visitor Rate:")
        top_return_urls = url_summary.nlargest(5, 'return_visitor_rate')[['page_visits', 'unique_visitors', 'return_visitor_rate']]
        print(top_return_urls)
        
        return url_summary
    
    def marketing_segment_analysis(self):
        """Analyze performance by marketing segment"""
        if self.df is None or 'marketing_segment' not in self.df.columns:
            print("Marketing segment data not available")
            return None
        
        print("\n" + "="*60)
        print("MARKETING SEGMENT ANALYSIS")
        print("="*60)
        
        # Aggregate by marketing segment
        segment_metrics = self.df.groupby('marketing_segment').agg({
            'page_visits': ['sum', 'mean', 'std'],
            'unique_visitors': ['sum', 'mean', 'std'],
            'return_visitors': ['sum', 'mean', 'std']
        }).round(2)
        
        # Calculate additional metrics
        segment_summary = self.df.groupby('marketing_segment').agg({
            'page_visits': 'sum',
            'unique_visitors': 'sum',
            'return_visitors': 'sum'
        })
        
        segment_summary['return_visitor_rate'] = (segment_summary['return_visitors'] / segment_summary['unique_visitors'] * 100).round(2)
        segment_summary['pages_per_visitor'] = (segment_summary['page_visits'] / segment_summary['unique_visitors']).round(2)
        segment_summary['traffic_share'] = (segment_summary['page_visits'] / segment_summary['page_visits'].sum() * 100).round(2)
        
        print("Marketing Segment Performance:")
        print(segment_summary.sort_values('page_visits', ascending=False))
        
        print(f"\nDetailed Marketing Segment Statistics:")
        print(segment_metrics)
        
        return segment_summary
        """Analyze performance by marketing segment"""
        if self.df is None or 'marketing_segment' not in self.df.columns:
            print("Marketing segment data not available")
            return None
        
        print("\n" + "="*60)
        print("MARKETING SEGMENT ANALYSIS")
        print("="*60)
        
        # Aggregate by marketing segment
        segment_metrics = self.df.groupby('marketing_segment').agg({
            'page_visits': ['sum', 'mean', 'std'],
            'unique_visitors': ['sum', 'mean', 'std'],
            'return_visitors': ['sum', 'mean', 'std']
        }).round(2)
        
        # Calculate additional metrics
        segment_summary = self.df.groupby('marketing_segment').agg({
            'page_visits': 'sum',
            'unique_visitors': 'sum',
            'return_visitors': 'sum'
        })
        
        segment_summary['return_visitor_rate'] = (segment_summary['return_visitors'] / segment_summary['unique_visitors'] * 100).round(2)
        segment_summary['pages_per_visitor'] = (segment_summary['page_visits'] / segment_summary['unique_visitors']).round(2)
        segment_summary['traffic_share'] = (segment_summary['page_visits'] / segment_summary['page_visits'].sum() * 100).round(2)
        
        print("Marketing Segment Performance:")
        print(segment_summary.sort_values('page_visits', ascending=False))
        
        print(f"\nDetailed Marketing Segment Statistics:")
        print(segment_metrics)
        
    def combined_analysis(self):
        """Analyze performance by URL and marketing segment combinations"""
        if self.df is None or 'url' not in self.df.columns or 'marketing_segment' not in self.df.columns:
            print("URL or marketing segment data not available")
            return None
        
        print("\n" + "="*60)
        print("URL + MARKETING SEGMENT COMBINED ANALYSIS")
        print("="*60)
        
        # Create combined analysis
        combined_metrics = self.df.groupby(['url', 'marketing_segment']).agg({
            'page_visits': 'sum',
            'unique_visitors': 'sum',
            'return_visitors': 'sum'
        }).reset_index()
        
        combined_metrics['return_visitor_rate'] = (combined_metrics['return_visitors'] / combined_metrics['unique_visitors'] * 100).round(2)
        combined_metrics['pages_per_visitor'] = (combined_metrics['page_visits'] / combined_metrics['unique_visitors']).round(2)
        
        print("Combined Performance Summary (Top 10 by Page Visits):")
        top_combinations = combined_metrics.nlargest(10, 'page_visits')
        print(top_combinations[['url', 'marketing_segment', 'page_visits', 'unique_visitors', 'return_visitor_rate']])
        
        # Best performing segment for each URL
        print(f"\nBest Marketing Segment for Each URL:")
        best_segments = combined_metrics.loc[combined_metrics.groupby('url')['page_visits'].idxmax()]
        print(best_segments[['url', 'marketing_segment', 'page_visits', 'unique_visitors']].head(10))
        
        return combined_metrics
    
    def time_series_analysis(self):
        """Analyze trends over time"""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print("TIME SERIES ANALYSIS")
        print("="*60)
        
        # Create monthly aggregation
        monthly_data = self.df.copy()
        monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
        
        monthly_metrics = monthly_data.groupby('year_month').agg({
            'page_visits': 'sum',
            'unique_visitors': 'sum',
            'return_visitors': 'sum'
        })
        
        # Calculate month-over-month growth
        monthly_metrics['visits_growth'] = monthly_metrics['page_visits'].pct_change() * 100
        monthly_metrics['unique_growth'] = monthly_metrics['unique_visitors'].pct_change() * 100
        
        print("Monthly Performance:")
        print(monthly_metrics.round(2))
        
        # Weekly aggregation
        weekly_data = self.df.copy()
        weekly_data['year_week'] = weekly_data['date'].dt.to_period('W')
        
        weekly_metrics = weekly_data.groupby('year_week').agg({
            'page_visits': 'sum',
            'unique_visitors': 'sum',
            'return_visitors': 'sum'
        })
        
        print(f"\nWeekly Trends (Last 10 weeks):")
        print(weekly_metrics.tail(10).round(2))
        
        return monthly_metrics, weekly_metrics
    
    def seasonal_analysis(self):
        """Analyze seasonal patterns"""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print("SEASONAL ANALYSIS")
        print("="*60)
        
        seasonal_data = self.df.copy()
        seasonal_data['month'] = seasonal_data['date'].dt.month
        seasonal_data['quarter'] = seasonal_data['date'].dt.quarter
        seasonal_data['day_of_year'] = seasonal_data['date'].dt.dayofyear
        
        # Monthly patterns
        monthly_patterns = seasonal_data.groupby('month').agg({
            'page_visits': ['mean', 'sum'],
            'unique_visitors': ['mean', 'sum'],
            'return_visitors': ['mean', 'sum']
        }).round(2)
        
        print("Monthly Patterns:")
        print(monthly_patterns)
        
        # Quarterly patterns
        quarterly_patterns = seasonal_data.groupby('quarter').agg({
            'page_visits': ['mean', 'sum'],
            'unique_visitors': ['mean', 'sum'],
            'return_visitors': ['mean', 'sum']
        }).round(2)
        
        print(f"\nQuarterly Patterns:")
        print(quarterly_patterns)
        
        return monthly_patterns, quarterly_patterns
    
    def performance_insights(self):
        """Generate key insights and recommendations"""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Overall performance metrics
        total_visits = self.df['page_visits'].sum()
        total_unique = self.df['unique_visitors'].sum()
        total_return = self.df['return_visitors'].sum()
        
        overall_return_rate = (total_return / total_unique * 100) if total_unique > 0 else 0
        avg_pages_per_visitor = total_visits / total_unique if total_unique > 0 else 0
        
        print(f"Overall Performance Summary:")
        print(f"• Total Page Visits: {total_visits:,}")
        print(f"• Total Unique Visitors: {total_unique:,}")
        print(f"• Total Return Visitors: {total_return:,}")
        print(f"• Overall Return Visitor Rate: {overall_return_rate:.2f}%")
        print(f"• Average Pages per Visitor: {avg_pages_per_visitor:.2f}")
        
        # Top performing marketing segments
        if 'marketing_segment' in self.df.columns:
            segment_performance = self.df.groupby('marketing_segment')['page_visits'].sum().sort_values(ascending=False)
            print(f"\nTop 3 Marketing Segments by Traffic:")
            for i, (segment, visits) in enumerate(segment_performance.head(3).items(), 1):
                percentage = (visits / total_visits * 100)
                print(f"{i}. {segment}: {visits:,} visits ({percentage:.1f}%)")
        
        # Date range insights
        date_range = (self.df['date'].max() - self.df['date'].min()).days
        avg_daily_visits = total_visits / date_range if date_range > 0 else 0
        
        print(f"\nTemporal Insights:")
        print(f"• Analysis Period: {date_range} days")
        print(f"• Average Daily Visits: {avg_daily_visits:.0f}")
        
        # Best and worst performing days
        daily_totals = self.df.groupby('date')['page_visits'].sum()
        best_day = daily_totals.idxmax()
        worst_day = daily_totals.idxmin()
        
        print(f"• Best Day: {best_day.strftime('%Y-%m-%d')} ({daily_totals[best_day]:,} visits)")
        print(f"• Worst Day: {worst_day.strftime('%Y-%m-%d')} ({daily_totals[worst_day]:,} visits)")
    
    def export_summary_report(self, output_filename=None):
        """Export a comprehensive summary report"""
        if self.df is None:
            return
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"analytics_summary_report_{timestamp}.csv"
        
        # Create summary datasets
        daily_summary = self.df.groupby('date').agg({
            'page_visits': 'sum',
            'unique_visitors': 'sum',
            'return_visitors': 'sum'
        }).reset_index()
        
        if 'marketing_segment' in self.df.columns:
            segment_summary = self.df.groupby('marketing_segment').agg({
                'page_visits': ['sum', 'mean'],
                'unique_visitors': ['sum', 'mean'],
                'return_visitors': ['sum', 'mean']
            })
            
        if 'url' in self.df.columns:
            url_summary = self.df.groupby('url').agg({
                'page_visits': ['sum', 'mean'],
                'unique_visitors': ['sum', 'mean'],
                'return_visitors': ['sum', 'mean']
            })
            
            # Save multiple summaries
            with pd.ExcelWriter(output_filename.replace('.csv', '.xlsx')) as writer:
                daily_summary.to_excel(writer, sheet_name='Daily_Summary', index=False)
                if 'marketing_segment' in self.df.columns:
                    segment_summary.to_excel(writer, sheet_name='Segment_Summary')
                if 'url' in self.df.columns:
                    url_summary.to_excel(writer, sheet_name='URL_Summary')
                self.df.to_excel(writer, sheet_name='Raw_Data', index=False)
        else:
            daily_summary.to_csv(output_filename, index=False)
        
        print(f"\nSummary report exported to: {output_filename}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        if self.df is None:
            print("No data available for analysis")
            return
        
        print("Starting comprehensive web analytics analysis...")
        print("="*80)
        
        # Run all analysis modules
        self.basic_statistics()
        daily_metrics = self.daily_analysis()
        url_metrics = self.url_analysis()
        segment_metrics = self.marketing_segment_analysis()
        combined_metrics = self.combined_analysis()
        monthly_metrics, weekly_metrics = self.time_series_analysis()
        seasonal_patterns = self.seasonal_analysis()
        self.performance_insights()
        
        # Export summary report
        self.export_summary_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return {
            'daily_metrics': daily_metrics,
            'url_metrics': url_metrics,
            'segment_metrics': segment_metrics,
            'combined_metrics': combined_metrics,
            'monthly_metrics': monthly_metrics,
            'weekly_metrics': weekly_metrics,
            'seasonal_patterns': seasonal_patterns
        }

# Usage example
if __name__ == "__main__":
    # Initialize analyzer with your data file
    filename = "web_analytics_data_2024-01-01_to_2025-07-31.csv"
    
    # Create analyzer instance
    analyzer = WebAnalyticsAnalyzer(filename)
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    # Optional: Access specific analysis results
    if results and results['daily_metrics'] is not None:
        print(f"\nQuick Access Examples:")
        print(f"Average daily page visits: {results['daily_metrics']['page_visits'].mean():.0f}")
        print(f"Peak daily page visits: {results['daily_metrics']['page_visits'].max():,}")
        
        if results['url_metrics'] is not None:
            top_url = results['url_metrics']['page_visits'].idxmax()
            print(f"Top performing URL: {top_url}")
            
        if results['segment_metrics'] is not None:
            top_segment = results['segment_metrics']['page_visits'].idxmax()
            print(f"Top marketing segment: {top_segment}")
    
    print(f"\n✓ Analysis complete! Check the generated Excel report for detailed results.")