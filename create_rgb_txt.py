# Read a csv file of the form
# Timestamp, filename.jpg
# and create a text file of the form
# data/filename.jpg
# for each line in the csv file

import csv
import sys
import os

csv_path = sys.argv[1]
outfile_path = os.path.join(os.path.dirname(csv_path), 'rgb.txt')
print("Saving to " + outfile_path)

outfile = open(outfile_path, 'w')

with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        outfile.write(row[0] + ' data/' + row[1] + '\n')
        
