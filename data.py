import pandas as pd
import re
import json
import os

def preprocess_chat(input_file, output_file):
    # Read the file into a DataFrame
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    df = pd.DataFrame(lines, columns=['text'])

    # Remove unnecessary lines
    df = df[~df['text'].str.contains('Messages and calls are end-to-end encrypted.', na=False)]

    # Extract timestamp, user, and message
    df['user'] = df['text'].apply(lambda x: re.search(r'- (.*?):', x).group(1) if re.search(r'- (.*?):', x) else None)
    df['message'] = df['text'].apply(lambda x: re.sub(r'^.*? - .*?: ', '', x))

    # Drop rows where user is None (i.e., system messages)
    df = df.dropna(subset=['user'])

    # Group and concatenate messages by user and their consecutive order
    grouped = df.groupby((df['user'] != df['user'].shift()).cumsum()).agg({'user': 'first', 'message': ' '.join}).reset_index(drop=True)

    # Convert to list of dictionaries
    result = grouped.apply(lambda row: {row['user']: row['message']}, axis=1).tolist()

    # Write JSON to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
