import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the CSV file into a pandas DataFrame
csv_path = "dataset/mtsamples/mtsamples.csv"
df = pd.read_csv(csv_path)

df = df.dropna()
label_encoder = LabelEncoder()

# Fit the LabelEncoder on the 'medical_specialty' column and transform the labels into integers
df['medical_specialty'] = label_encoder.fit_transform(df['medical_specialty'])


# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the training and validation sets to CSV files
train_df.to_csv("dataset/mtsamples/train.csv", index=False)
val_df.to_csv("dataset/mtsamples/test.csv", index=False)

print("Training set size:", len(train_df))
print("Validation set size:", len(val_df))