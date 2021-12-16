import pandas as pd, numpy as np
import datetime, random

from faker import Faker
from pandas.core.base import DataError
fake = Faker()

name = []
subjects = ["MATH","LIT","ENG","BIO","CHEM","PHYS"]
genders = ["M","F"]
startTime = []
endTime = []

print ('How many users to generate ')
n_names = int(input())

for n in range(n_names):
    name.append(fake.name())
    time = fake.unix_time(datetime.datetime(2021, 12, 31))
    startTime.append(time)
    endTime.append(time + random.randrange(1,3,1)*3600)

subject = np.random.choice(subjects, n_names, p=[.25,.25,.2,.1,.1,.1])
gender = np.random.choice(genders, n_names, p=[.45,.55])

variables = [name, subject, gender, startTime, endTime]

df = pd.DataFrame(variables).transpose()

df.columns = ["Name","Subject","Gender","StartTime","EndTime"]

df.to_csv('test.csv')

