import pandas as pd

columns = {"TEXT": "text", "Label": 'label', 'id': 'id'}

# Ensure all the labels are in mappings

emoji_mapping = pd.read_csv('data/raw/Mapping.csv')
train_data = pd.read_csv('data/raw/Train.csv', usecols=['TEXT', 'Label'])
test_data = pd.read_csv('data/raw/Test.csv', usecols=['TEXT', 'id'])

emojis = emoji_mapping['number']
assert (all(train_data['Label'].isin(emojis)))

def clean_data(df):
    print (df.columns)

    df['TEXT'] = df['TEXT'].str.strip("\n")
    df = df.rename(columns=columns)
    return df

train_data = clean_data(train_data)
test_data = clean_data(test_data)
train_data.to_csv('data/processed/train.csv', header=True, index=False)
test_data.to_csv('data/processed/test.csv',header=True, index=False)