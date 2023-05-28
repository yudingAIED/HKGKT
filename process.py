# -*- coding: utf-8 -*-
"""
Created on Sat May 13 09:42:27 2023

@author: wsxok
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random

from config import *

def save(file, var):
    np.save(file, var)
    print('save ' + file)

def parse_all_seq(data, students):
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence:\t'):
        student_sequence = parse_student_seq(data[data.student_id == student_id])
        all_sequences.extend([student_sequence])
    return all_sequences

def parse_student_seq(student):
    seq = student.sort_values('order_id')
    q = [q for q in seq.question_id.tolist()]
    a = seq.correct.tolist()
    return q, a


def data_process(file, edge_index, edge_type, max_length):
    data = pd.read_csv(
    file,
    usecols=['student_id', 'question_id', 'concept', 'score', 'full_score'], 
    encoding='unicode_escape').dropna(subset=['full_score'])
    
    data['score'] = data['score'].astype(int)
    data['full_score'] = data['full_score'].astype(int)
    
    data.loc[:,'correct'] = data['score'] / data['full_score']
    data['correct'] = data['correct'].astype(int)
    del data['score']
    del data['full_score']
    data.loc[:,'order_id'] = 0
    
    student_query = dict()
    question_query = dict()
    concept_query = dict()
    t1 = 0
    t2 = 0
    t3 = 0

    for i in range(0,data.shape[0]):
        if data.iloc[i,0] not in student_query.keys():
            student_query[data.iloc[i,0]] = t1
            t1 += 1
        if data.iloc[i,1] not in question_query.keys():
            question_query[data.iloc[i,1]] = t2
            t2 += 1
        if data.iloc[i,2] not in concept_query.keys():
            concept_query[data.iloc[i,2]] = t3
            t3 += 1
    
    for i in range(0,data.shape[0]):
        data.iloc[i,0] = student_query[data.iloc[i,0]]
        data.iloc[i,1] = question_query[data.iloc[i,1]]
        data.iloc[i,2] = concept_query[data.iloc[i,2]]
        data.iloc[i,4] = i
    
    Q_info = [t3 for x in range(t2 + 1)]
    try_total = [0 for x in range(t2)]
    correct_total = [0 for x in range(t2)]
    
    for i in range(0,data.shape[0]):
        Q_info[data.iloc[i,1]] = data.iloc[i,2]
        try_total[data.iloc[i,1]] += 1
        if data.iloc[i,3] >= 1:
            correct_total[data.iloc[i,1]] += 1
    correct_rate = []
    for i in range(len(try_total)):
        correct_rate.append(correct_total[i]/try_total[i])
    
    diff = np.ones(len(correct_rate) + 1, dtype=np.int16) * 5    #5-difficulty-level
    for i in range(0,len(correct_rate)):
        if correct_rate[i]<=0.2:
            diff[i] = 4
        elif correct_rate[i]>0.2 and correct_rate[i]<=0.4:
            diff[i] = 3
        elif correct_rate[i]>0.4 and correct_rate[i]<=0.6:
            diff[i] = 2
        elif correct_rate[i]>0.6 and correct_rate[i]<=0.8:
            diff[i] = 1
        else:
            diff[i] = 0
    
    sequences = parse_all_seq(data, data.student_id.unique())
    
    q = np.ones((len(sequences),max_length), dtype=np.int16) * (t2)
    a = np.ones((len(sequences),max_length), dtype=np.int16) * 2
    
    for i in range(0,len(sequences)):
        length = min(max_length,len(sequences[i][0]))
        for j in range(0,length):
            q[i][j] = sequences[i][0][j]
            a[i][j] = sequences[i][1][j]
    
    Q_info = np.array(Q_info)
    
    save("Q_info.npy", Q_info)
    save("diff.npy", diff)
    save("q.npy", q)
    save("a.npy", a)
    save("edge_index.npy", edge_index)
    save("edge_type.npy", edge_type)

    print("processing done!")
    
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            