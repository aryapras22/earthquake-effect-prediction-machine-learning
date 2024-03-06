data2 = df.drop(columns=['floors_before_eq (total)'])

import re

def convert_to_int(string):
    if isinstance(string, str):
        match = re.search(r'\d+', string)
        if match:
            numeric_part = match.group(0)
            return int(numeric_part)
    return string

df['floors_before_eq (total)'] = df['floors_before_eq (total)'].apply(convert_to_int)

print(df)