import json
import csv
import ast

# Opening JSON file and loading the data into the variable data 
data=[] 
with open('qa_Electronics.json', encoding= 'utf-8') as json_file:
    for i in json_file:
        c= ast.literal_eval(i) 
        data.append(c)

data_file= open('data_file.csv','w')


# create the csv writer object 
csv_writer = csv.writer(data_file)

#Counter variable used for writing headers to the CSV file

header  = ['questionType', 'asin', 'answerTime' , 'unixTime' , 'question' , 'answerType', "answer"]
csv_writer.writerow(header)

for emp in data:
    default_list=['NaN', 'NaN', 'NAN', 'NaN','NaN','NaN' ,'NaN']
    if len(emp)!=0:
        for j in emp:
            if j == 'questionType':
                default_list[0]=emp[j]
            if j == 'asin':
                default_list[1]=emp[j]
            if j == 'answerTime':
                default_list[2]=emp[j]
            if j == 'unixTime':
                default_list[3]=emp[j]
            if j == 'question':
                default_list[4]=emp[j] 
            if j == 'answerType':
                default_list[5]=emp[j]
            if j == 'answer':
                default_list[6]=emp[j]
            
        csv_writer.writerow(default_list)
data_file.close()





