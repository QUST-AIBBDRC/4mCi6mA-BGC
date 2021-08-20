import os
import sys,re
import pickle
from collections import Counter
from functools import reduce
import itertools
import pandas as pd
import numpy as np
import platform

ALPHABET='ACGT'

def readDNAFasta(file):
	with open(file) as f:
		records = f.read()
	if re.search('>', records) == None:
		print('Error,the input DNA sequence must be fasta format.')
		sys.exit(1)
	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ACGT-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta

	
def binary(input_data):
    fastas=readDNAFasta(input_data)
    vector = []
    header = ['#']
    for i in range(1, len(fastas[0][1]) * 4 + 1):
        header.append('BINARY.F'+str(i))
    vector.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for aa in sequence:
            if aa == '-':
                code = code + [0, 0, 0, 0]
                continue
            for aa1 in ALPHABET:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
        vector.append(code)
    return vector
    
input_vector=binary('Sample.txt')