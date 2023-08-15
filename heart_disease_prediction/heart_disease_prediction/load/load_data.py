import pandas as pd
import numpy as np
import re
import os

class DataRetriever:
    """A class for retrieving data from a CSV file."""

    def __init__(self, filepath):
        """Initializes the DataRetriever with the given filepath."""
        self.filepath = filepath

    def read_data(self):
        """Reads the data from the CSV file and returns a Pandas DataFrame."""
        df = pd.read_csv(self.filepath)
        return df
