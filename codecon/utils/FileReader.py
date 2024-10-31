
import pandas as pd
import os

class FileReader:
    def __init__(self, data_pred=None, data_raw=None):
        self.data_pred = data_pred
        self.data_raw = data_raw
        self.df_pred = None
        self.df_raw = None

    def read_pred_file(self):
        """Reads the prediction data file and checks for 'text' column."""
        if isinstance(self.data_pred, str):
            self.data_pred = os.path.normpath(self.data_pred)

            if self.data_pred.endswith('.csv'):
                self.df_pred = pd.read_csv(self.data_pred)
            elif self.data_pred.endswith(('.xlsx', '.xls')):
                self.df_pred = pd.read_excel(self.data_pred)
            else:
                raise ValueError("Unsupported file format for data_pred. Please use .csv or .excel files.")
        else:
            raise ValueError("The data_pred file path is not a string")

        # Ensure 'text' column exists in data_pred
        if 'text' not in self.df_pred.columns:
            raise ValueError("Data_pred must contain a 'text' column.")

        return self.df_pred

    def read_raw_file(self):
        """Reads the raw data file and checks for 'text' and 'label' columns."""
        if isinstance(self.data_raw, str):
            self.data_raw = os.path.normpath(self.data_raw)

            if self.data_raw.endswith('.csv'):
                self.df_raw = pd.read_csv(self.data_raw)
            elif self.data_raw.endswith(('.xlsx', '.xls')):
                self.df_raw = pd.read_excel(self.data_raw)
            else:
                raise ValueError("Unsupported file format for data_raw. Please use .csv or .excel files.")
        else:
            raise ValueError("The data_raw file path is not a string")

        # Ensure 'text' and 'label' columns exist in data_raw
        if 'text' not in self.df_raw.columns or 'label' not in self.df_raw.columns:
            raise ValueError("Data_raw must contain 'text' and 'label' columns.")

        labels = self.df_raw['label'].astype(int).tolist()
        unique_labels = sorted(list(set(labels)))
        # Check if labels are integers and start from 0
        if not all(isinstance(label, int) for label in unique_labels):
            raise ValueError("Labels must be integers")
        if unique_labels != list(range(len(unique_labels))):
            raise ValueError(f"[WARNING] Labels are not consecutive or do not start from 0")

        print(f"Labels: {sorted(list(set(labels)))}")
        return self.df_raw
