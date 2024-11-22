import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def create_interactive_plot(file_path, start_date=None, end_date=None):
    """
    Create an interactive time series plot with date range selection and line isolation.
    
    Parameters:
    file_path (str): Path to the CSV file
    start_date (str): Optional start date in 'YYYY-MM-DD' format
    end_date (str): Optional end date in 'YYYY-MM-DD' format
    """
    # Read the CSV file with the index column as dates
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Sort index to ensure proper plotting
    df = df.sort_index()
    
    # Apply date filters if provided
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces for each column
    for column in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                name=column,
                mode='lines+markers',
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=(
                    f"<b>{column}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Value: %{y:.2f}<br>"
                    "<extra></extra>"
                )
            )
        )
    
    # Update layout with interactive features
    fig.update_layout(
        title={
            'text': 'Interactive Time Series Data',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis=dict(
            title='Date',
            rangeslider=dict(visible=True),  # Add range slider
            rangeselector=dict(  # Add range selector buttons
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        ),
        yaxis=dict(title='Value'),
        hovermode='x unified',  # Show all values for a given x-coordinate
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        width=1200,
        height=800,
        margin=dict(r=150)  # Add right margin for legend
    )
    
    return fig

def save_and_show_plot(fig, output_path=None):
    """
    Save and display the interactive plot.
    
    Parameters:
    fig: Plotly figure object
    output_path (str): Optional path to save the HTML file
    """
    if output_path:
        fig.write_html(output_path)
    fig.show()

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = 'FamaFrenchData.csv'
    
    # Create and display the plot
    fig = create_interactive_plot(
        file_path,
        start_date='2007-01-01',  # Optional: specify date range
        end_date='2010-01-01'     # Optional: specify date range
    )
    
    # Save and show the plot (optional)
    save_and_show_plot(fig, 'interactive_plot.html')