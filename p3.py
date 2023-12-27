import pandas as pd
df=pd.read_csv('salaries.csv')
print("First few rows")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
filtered_data=df[df['Age']>30]
print("\n Filtered data (Age>30):")
print(filtered_data)
sorted_data=df.sort_values(by='Salary',ascending=False)
print("\n Sorted data(by Salary):")
print(sorted_data)
df['Bonus']=df['Salary']*0.1
print("\n Data with new column (Bonus)")
print(df)
df.to_csv("output.csv",index=False)
print("\n data written to output.csv")
'''
First few rows
              Name  Age                  Job  Salary
0    Kevin Sanders   24    Software Engineer    7300
1       Lisa Mills   26  High School Teacher    6100
2    Donna Allison   27              Dentist   12700
3  Michael Schmitt   43              Dentist   17500
4     Lisa Shaffer   31           Accountant    7400

Summary statistics:
               Age        Salary
count  1000.000000   1000.000000
mean     43.241000  13609.500000
std      12.485784   4242.159316
min      23.000000   4000.000000
25%      32.000000  10300.000000
50%      43.000000  13300.000000
75%      54.000000  16800.000000
max      65.000000  24100.000000

 Filtered data (Age>30):
                Name  Age                  Job  Salary
3    Michael Schmitt   43              Dentist   17500
4       Lisa Shaffer   31           Accountant    7400
6       Joanne Perez   52    Software Engineer   17200
7     Jeffrey Wilson   56              Dentist   20400
9    Frank Gutierrez   46           Accountant   12400
..               ...  ...                  ...     ...
994   Rebecca Becker   42  High School Teacher   12200
995   Darin Erickson   47           Accountant   13200
996   Scott Mcdaniel   59  High School Teacher   17300
997      Erica Smith   35  High School Teacher    8100
998      Tanya Jones   54              Dentist   20300

[792 rows x 4 columns]

 Sorted data(by Salary):
                    Name  Age                  Job  Salary
262          Jacob Moran   65              Dentist   24100
741        Scott Russell   65              Dentist   23600
873        Tyler Bennett   65              Dentist   23600
901      Jessica Jackson   63              Dentist   23500
292   Elizabeth Buchanan   65              Dentist   23100
..                   ...  ...                  ...     ...
458      Sarah Zimmerman   23  High School Teacher    5000
110    Brittney Stephens   25  High School Teacher    4600
369          Carl Franco   25  High School Teacher    4600
557        Roberto Bryan   24  High School Teacher    4300
968  Dr. Angela Wells MD   23  High School Teacher    4000

[1000 rows x 4 columns]

 Data with new column (Bonus)
                Name  Age                  Job  Salary   Bonus
0      Kevin Sanders   24    Software Engineer    7300   730.0
1         Lisa Mills   26  High School Teacher    6100   610.0
2      Donna Allison   27              Dentist   12700  1270.0
3    Michael Schmitt   43              Dentist   17500  1750.0
4       Lisa Shaffer   31           Accountant    7400   740.0
..               ...  ...                  ...     ...     ...
995   Darin Erickson   47           Accountant   13200  1320.0
996   Scott Mcdaniel   59  High School Teacher   17300  1730.0
997      Erica Smith   35  High School Teacher    8100   810.0
998      Tanya Jones   54              Dentist   20300  2030.0
999        Mark Knox   25  High School Teacher    6600   660.0

[1000 rows x 5 columns]

 data written to output.csv
 '''
