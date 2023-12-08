import pandas as pd
import plotly.express as px
from typing import Union

class Visualizer:
    """
    A class for visualizing the count of labels in a DataFrame using Plotly Express.
    """

    @staticmethod
    def show_count_of_labels(df: pd.DataFrame, x: str = 'label') -> None:
        """
        Display a histogram showing the count of labels in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data_handlers.
            x (str): The column name for which the histogram is to be displayed.
        """
        # Create a histogram using Plotly Express
        fig = px.histogram(df,
                           x=x,  # The column in the DataFrame
                           title=f'Histogram of {x}',
                           template='ggplot2',  # Using the ggplot2 template for a clean design
                           color=x,  # Color the histogram bars based on the specified column
                           color_discrete_sequence=px.colors.sequential.Blues_r,  # Color sequence for better visibility
                           opacity=0.8,  # Set opacity for better visualization
                           height=525,  # Set the height of the plot
                           width=835,  # Set the width of the plot
                           )

        # Update the y-axis title
        fig.update_yaxes(title='Count')

        # Show the generated plot
        fig.show()
