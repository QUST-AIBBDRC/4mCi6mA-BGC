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
	
def kmerArray(sequence, k):
     kmer = []
     for i in range(len(sequence) - k + 1):
         kmer.append(sequence[i:i + k])
     return kmer
	
def Kmer(input_data, k=1, normalize=True):
    
    fastas=readDNAFasta(input_data)
    vector = []
    header = ['#']
    if k < 1:
        print('error, the k must be positive integer.')
        return 0
    for kmer in itertools.product(ALPHABET, repeat=k):
        header.append(''.join(kmer))
    vector.append(header)
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        kmers = kmerArray(sequence, k)
        count = Counter()
        count.update(kmers)
        if normalize == True:
           for key in count:
               count[key] = count[key] / len(kmers)
        code = [name]
        for j in range(1, len(header)):
            if header[j] in count:
               code.append(count[header[j]])
            else:
                code.append(0)
        vector.append(code)
    return vector

vector=Kmer('Sample.txt',k)
