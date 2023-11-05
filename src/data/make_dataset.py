import pandas as pd

# Reading the raw tsv file
df = pd.read_csv("../../data/raw/filtered.tsv", sep="\t")


# Making the train dataset for model training with one-to-one toxic sentence and 
# corresponding detoxified sentences in a dataframe
non_toxic, toxic = [],[]
for rows, columns in df.iterrows():
  if(columns['ref_tox'] < 0.5 and columns['trn_tox'] > 0.5):
    non_toxic.append(columns['reference'])
    toxic.append(columns['translation'])

  elif(columns['ref_tox'] > 0.5 and columns['trn_tox'] < 0.5):
    toxic.append(columns['reference'])
    non_toxic.append(columns['translation'])

# storing in the new df
new_df = pd.DataFrame()
new_df["toxic"] = toxic
new_df["detoxified"] = non_toxic    

# saving it in the interim folder
new_df.to_csv("../../data/interim/detoxification_dataset.csv", index=False)