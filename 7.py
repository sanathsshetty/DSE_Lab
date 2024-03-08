import pandas as pd

# Create DataFrame
data = {'Name': ['John', 'Emma', 'Sant', 'Lisa', 'Tom'],
        'Age': [25, 30, 28, 32, 27],
        'Country': ['USA', 'Canada', 'India', 'UK', 'Australia'],
        'Salary': [50000, 60000, 70000, 80000, 65000]}
df = pd.DataFrame(data)

# Original DataFrame
print("Original DataFrame:")
print(df)

# Selecting specific columns
name_age = df[['Name', 'Age']]
print("\nName and Age columns:")
print(name_age)

# Filtering DataFrame
filtered_df = df[df['Country'] == 'USA']
print("\nFiltered DataFrame (Country='USA'):")
print(filtered_df)

# Sorting DataFrame
sorted_df = df.sort_values("Salary", ascending=False)
print("\nSorted DataFrame (by salary in descending order):")
print(sorted_df)

# Calculating average salary
average_salary = df['Salary'].mean()
print("\nAverage salary:", average_salary)

# Adding a new column
df['Experience'] = [3, 6, 4, 8, 5]
print("\nDataFrame with added experience:")
print(df)

# Updating salary for a specific row
df.loc[df['Name'] == 'Emma', 'Salary'] = 65000
print("\nDataFrame with updated Emma's salary:")
print(df)

# Dropping a column
df = df.drop('Experience', axis=1)
print("\nDataFrame after deleting the column:")
print(df)
