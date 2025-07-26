import pandas as pd

# Load the Excel file using the relative path from your script location
df = pd.read_excel('../experimentFiles/xab.xlsx')


# Define a function to determine 'within' or 'between' based on columns A and B
def get_within_between(row):
    # Get last 9 characters of column A and B, handling possible NaN values
    a_last = str(row['A'])[-9:] if pd.notna(row['A']) else ''
    b_last = str(row['B'])[-9:] if pd.notna(row['B']) else ''

    # If both are cat_0.jpg or both are cat_1.jpg, label as 'within'
    if a_last == 'cat_0.jpg' and b_last == 'cat_0.jpg':
        return 'within'
    elif a_last == 'cat_1.jpg' and b_last == 'cat_1.jpg':
        return 'within'
    else:
        # Otherwise, label as 'between'
        return 'between'


# Define a function to set the category for 'within' cases
def get_category(row):
    # Get last 9 characters of column A and B
    a_last = str(row['A'])[-9:] if pd.notna(row['A']) else ''
    b_last = str(row['B'])[-9:] if pd.notna(row['B']) else ''

    # If both are cat_0.jpg, set category as 'cat_0'
    if a_last == 'cat_0.jpg' and b_last == 'cat_0.jpg':
        return 'cat_0'
    # If both are cat_1.jpg, set category as 'cat_1'
    elif a_last == 'cat_1.jpg' and b_last == 'cat_1.jpg':
        return 'cat_1'
    else:
        # Otherwise, leave category blank
        return ''


# Apply the functions to create new columns
df['within_between'] = df.apply(get_within_between, axis=1)
df['category'] = df.apply(get_category, axis=1)

# Save the modified dataframe to a new CSV file (no index column in CSV)
df.to_csv('../experimentFiles/xab_within_between.csv', index=False)
