import pandas as pd
import re

class DataCleaner:
    def clean_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.lower().strip()  # Convert to lowercase

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize column names (lowercase, strip spaces)
        df.columns = df.columns.str.strip().str.lower()

        # Check that required columns exist
        if 'text' not in df.columns or 'biased' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'biased' columns.")

        # Keep only labeled tweets
        df = df[df['biased'].isin([0, 1])]

        # Clean the text
        df['text'] = df['text'].astype(str).apply(self.clean_text)

        return df