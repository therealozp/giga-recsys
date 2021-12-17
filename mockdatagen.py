import pandas as pd, numpy as np
import datetime, random

from faker import Faker

fake = Faker()

menteeName = []
mentorName = []
subjects = ["MATH","LIT","ENG","BIO","CHEM","PHYS"]
genders = ["M","F"]
mentorStartTime = []
mentorEndTime = []
menteeStartTime = []
menteeEndTime = []
countMatch = countNoMatch = m = t = 0
rating = []

print('How many matchings to generate')
n_names = int(input())
print('Name the file (no end tag)')
filename = input()

for n in range(n_names):
    mentorName.append(fake.name())
    menteeName.append(fake.name())
    timeMentor = fake.unix_time(datetime.datetime(2021, 12, 31))
    timeMentee = fake.unix_time(datetime.datetime(2021, 12, 31))
    mentorStartTime.append(timeMentor)
    mentorEndTime.append(timeMentor + random.randrange(1,3,1)*3600)
    menteeStartTime.append(timeMentee)
    menteeEndTime.append(timeMentee + random.randrange(1,3,1)*3600)

mentorSubject = np.random.choice(subjects, n_names, p=[.25,.25,.2,.1,.1,.1])
menteeSubject = np.random.choice(subjects, n_names, p=[.2,.2,.2,.2,.1,.1])
mentorGender = np.random.choice(genders, n_names, p=[.45,.55])
menteeGender = np.random.choice(genders, n_names, p=[.55,.45])

for i in range(n_names):
    if mentorSubject[i] == menteeSubject[i]:
        countMatch += 1
    else:
        countNoMatch += 1

highRatings = np.random.choice([2,3,4,5], countMatch, p=[.1,.1,.5,.3])
lowRatings = np.random.choice([1,2,3,4], countNoMatch, p=[.3,.5,.1,.1])

for i in range(n_names):
    if mentorSubject[i] == menteeSubject[i]:
        rating.append(highRatings[m])
        m += 1
    else:
        rating.append(lowRatings[t])
        n += 1

variables = [mentorName, mentorSubject, mentorGender, mentorStartTime, mentorEndTime, menteeName, menteeSubject, menteeGender, menteeStartTime, menteeEndTime, rating]

df = pd.DataFrame(variables).transpose()

df.columns = ["MentorName","MentorSubject","MentorGender","MentorStartTime","MentorEndTime","MenteeName","MenteeSubject","MenteeGender","MenteeStartTime","MenteeEndTime", "Rating"]

df.to_csv(filename+'.csv')

