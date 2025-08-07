import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime

def create_visits_line_graph(csv_file_path):
    """
    Function to create a line graph of visits over time from CSV data.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        plotly.graph_objs.Figure: Line graph figure
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Aggregate page visits by date (sum all visits for each day)
    daily_visits = df.groupby('date')['page_visits'].sum().reset_index()
    
    # Sort by date to ensure proper line plotting
    daily_visits = daily_visits.sort_values('date')
    
    # Create the line graph
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_visits['date'],
        y=daily_visits['page_visits'],
        mode='lines+markers',
        name='Page Visits',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4, color='#2E86AB'),
        hovertemplate='<b>Date:</b> %{x}<br>' +
                      '<b>Total Visits:</b> %{y:,}<br>' +
                      '<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Website Visits Over Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        xaxis=dict(
            title='Date',
            titlefont={'size': 14, 'color': '#2C3E50'},
            tickfont={'size': 12, 'color': '#2C3E50'},
            showgrid=True,
            gridcolor='#ECF0F1'
        ),
        yaxis=dict(
            title='Total Page Visits',
            titlefont={'size': 14, 'color': '#2C3E50'},
            tickfont={'size': 12, 'color': '#2C3E50'},
            showgrid=True,
            gridcolor='#ECF0F1'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    return fig

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.Div([
        html.H1("Website Analytics Dashboard", 
                style={
                    'textAlign': 'center',
                    'color': '#2C3E50',
                    'marginBottom': '30px',
                    'fontFamily': 'Arial, sans-serif'
                }),
        
        # Line graph component
        dcc.Graph(
            id='visits-line-graph',
            figure=create_visits_line_graph('your_file.csv'),
            style={'height': '600px'}
        )
    ], style={
        'padding': '20px',
        'maxWidth': '1200px',
        'margin': '0 auto',
        'fontFamily': 'Arial, sans-serif'
    })
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)