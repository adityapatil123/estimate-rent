import pandas as pd


def clean_rent_dataset_csv(existing_csv_path, new_csv_path):
    df = pd.read_csv(existing_csv_path)

    # Update Floor into 'Total Floors' and 'Floor'
    df['Total Floors'] =  df['Floor'].apply(lambda x: x.split("of ")[-1])
    df['Floor'] = df['Floor'].apply(lambda x: x.split(" out")[0])

    # Update Tenant Preferred into 'Bachelors Preferred' and 'Family Preferred'
    df['Bachelors Preferred'] = df['Tenant Preferred'].apply(lambda x: 'Bachelors' in x)
    df['Family Preferred'] = df['Tenant Preferred'].apply(lambda x: 'Family' in x)
    df = df.drop(['Tenant Preferred'], axis=1)

    # Convert dataframe to CSV
    df.to_csv(new_csv_path, sep=',', index=False)


if __name__ == "__main__":
    clean_rent_dataset_csv("dataset/House_Rent_Dataset.csv", "dataset/Cleaned_House_Rent_Dataset.csv")
