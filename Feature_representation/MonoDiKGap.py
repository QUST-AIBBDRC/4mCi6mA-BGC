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

def MonoDiKGap_vector(input_data,g):   
    fastas=readDNAFasta(input_data)
    vector=[] 
    header=['#']
    for f in range((g)*32):
        header.append('MonoDi.'+str(f))
    vector.append(header)
    sample=[]
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        sample = [name]
        each_vec=MonoDiKGap(sequence,g)
        sample=sample+each_vec
        vector.append(sample)
    return vector

vector=MonoDiKGap_vector('Sample.txt',g)
