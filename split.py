#!/usr/bin/python
import random

dataset = open('data.tsv', 'r')
trainingdata = open('trainingdata2.tsv', 'w')
validationdata = open('validationdata2.tsv', 'w')

line = dataset.readline();

#extracts 10% of the entire dataset and splits them into training and validation datasets(70:30 ratio)
#Ten lines in the dataset makes up one data point,so read and write in batches of 10 lines
while True:
    split_value = random.randint(1, 100)
    if split_value % 10 == 0:
    	
    	if split_value % 3 == 0:
        	write_file = validationdata
    	else:
        	write_file = trainingdata

    	write_file.write(line)
    	for i in range(9):
        	next_line = dataset.readline()
        	write_file.write(next_line)
    
    	if split_value % 3 == 0:
       		validationdata = write_file
    	elif split_value:
       		trainingdata = write_file
    else:
    	#skipped;
    	for i in range(9):
    		dataset.readline();
    
    line = dataset.readline();
    if line == "":#means eof file has been reached
        break
    
