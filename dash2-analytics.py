import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import os

# Initialize the Dash app
app = dash.Dash(__name__)

def create_visits_line_graph(df):
    """
    Function to create a line graph of visits over time
    
    Args:
        df (pandas.DataFrame): DataFrame containing date and visits data
    
    Returns:
        plotly.graph_objects.Figure: Line graph figure
    """
    # Try to identify date and visits columns automatically
    date_columns = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['date', 'time', 'day', 'month', 'year'])]
    
    visits_columns = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['visit', 'traffic', 'view', 'session', 'user', 'count'])]
    
    # Use the first matching columns, or fallback to first two columns
    date_col = date_columns[0] if date_columns else df.columns[0]
    visits_col = visits_columns[0] if visits_columns else df.columns[1]
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Ensure visits column is numeric (convert strings to numbers)
    df[visits_col] = pd.to_numeric(df[visits_col], errors='coerce')
    
    # Sort by date
    df = df.sort_values(date_col)
    
    # Create the line graph
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[visits_col],
        mode='lines+markers',
        name='Visits',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6, color='#1f77b4'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Visits:</b> %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Website Visits Over Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Date',
        yaxis_title='Number of Visits',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    # Format y-axis to show comma-separated numbers
    fig.update_yaxes(tickformat=',')
    
    return fig

# Load the data directly
df = pd.read_csv('web_analytics_data_2024-01-01_to_2025-07-31.csv')
data_df = df
data_source = 'web_analytics_data_2024-01-01_to_2025-07-31.csv'

# Ensure the visits column is numeric for calculations
visits_col_index = 1
if len(data_df.columns) > 1:
    data_df.iloc[:, visits_col_index] = pd.to_numeric(data_df.iloc[:, visits_col_index], errors='coerce')

# Create the app layout
app.layout = html.Div([
    html.Div([
        html.H1(
            "Website Traffic Dashboard", 
            style={
                'textAlign': 'center',
                'marginBottom': '30px',
                'color': '#2c3e50',
                'fontFamily': 'Arial, sans-serif'
            }
        ),
        
        html.P(
            f"Data source: {data_source}",
            style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'fontSize': '14px',
                'marginBottom': '20px'
            }
        ),
        
        # Line graph component
        dcc.Graph(
            id='visits-line-graph',
            figure=create_visits_line_graph(data_df),
            style={'marginBottom': '20px'}
        ),
        
        # Data summary
        html.Div([
            html.H3("Data Summary", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.P(f"Total data points: {len(data_df):,}", style={'margin': '5px 0'}),
            html.P(f"Date range: {data_df.iloc[:, 0].min()} to {data_df.iloc[:, 0].max()}", style={'margin': '5px 0'}),
            html.P(f"Total visits: {data_df.iloc[:, 1].sum():,.0f}", style={'margin': '5px 0'}),
            html.P(f"Average daily visits: {data_df.iloc[:, 1].mean():,.0f}", style={'margin': '5px 0'}),
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '5px',
            'margin': '20px 0'
        })
        
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif'
    })
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)