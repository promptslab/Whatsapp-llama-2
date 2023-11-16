import pandas as pd
import re
import json
import os

def preprocess_and_convert_to_samsum(input_file, output_csv):
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

    # Format context for Samsum
    def format_context(messages):
        context = ''
        for message in messages:
            user = list(message.keys())[0]
            context += f"{user}: {message[user]}\n"
        return context

    def format_output(message):
        return list(message.values())[0]

    # Convert the chat JSONs into the Samsum dataset format
    conv = []
    for count, message in enumerate(result):
        if count != 0:
            context = format_context(result[max(0, count-5):count])
            conv.append([context, format_output(message)])

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(conv, columns=['Context', 'Reply'])
    df.to_csv(output_csv, index=False)


class ChatDatasetFormatter:
    def __init__(self, file_name, user_name, save_file):
        self.file_name = file_name
        self.user_name = user_name
        self.save_file = save_file

    @staticmethod
    def format_context(messages):
        context = ''
        for message in messages:
            user = list(message.keys())[0]
            context += f"{user}: {message[user]}\n"
        return context

    @staticmethod
    def format_output(message):
        user = list(message.keys())[0]
        return message[user]

    def prepare_dataset(self):
        conversations = []
        try:
            with open(self.file_name, 'r') as f:
                data = json.load(f)

            print(f"Total messages in file: {len(data)}")  # Print total messages

            count = 0
            for message in data:
                if list(message.keys())[0] == self.user_name and count != 0:
                    context = self.format_context(data[max(0, count-5):count])
                    reply = self.format_output(message)
                    conversations.append([context, reply])

                count += 1

            if not conversations:
                print("No conversations found for the specified user.")

            df = pd.DataFrame(conversations, columns=['Context', 'Reply'])
            df.to_csv(self.save_file)
            print(f"CSV saved with {len(df)} conversations.")  # Print the number of conversations saved
        except json.JSONDecodeError as e:
            print(f"Error processing file {self.file_name}: {e}")
