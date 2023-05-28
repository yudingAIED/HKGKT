# -*- coding: utf-8 -*-
"""
Created on Sat May 13 09:28:34 2023

@author: wsxok
"""

import numpy as np

subject = 'phy'
file = 'data/unit-' + subject +'.csv'

edge_index_phy = np.array([[6,37,3,42,28,28,41,7,44,44,44,44,44,1,26,
                        3,20,19,23,3,5,24,30,2,15,3,12,5,14,29,12,5,44,
                        13,13,9,31,16,16,18,17,17,43,47,19,19,12,44,12,12],
                        [7,8,2,28,35,45,48,6,7,37,36,34,6,26,4,36,21,23,33,24,24,32,
                        31,15,38,14,9,0,3,45,29,29,29,10,15,31,30,17,18,16,16,20,
                        40,46,11,27,28,12,29,32]])
edge_type_phy = np.array([1,1,0,1,1,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0,
                     1,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,1])

edge_index_math = np.array([[0,0,0,1,1,1,1,3,4,5,6,7,7,10,11,13,14,15,15,15,16,16,
                       17,18,21,21,22,23,25,28,29],
                        [3,10,4,15,20,26,32,10,15,8,14,12,11,3,7,14,13,19,21,22,17,28,
                        29,13,17,29,25,28,22,23,17]])
edge_type_math = np.array([0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,1,1,1,1])

edge_index_chem = np.array([[1,2,2,3,3,3,3,3,5,5,5,5,10,13,14,14,16],
                        [3,12,17,2,14,17,6,1,6,15,4,14,6,16,9,5,13]])
edge_type_chem = np.array([1,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,1])


edge_index_dict = {'phy': edge_index_phy,
                   'math': edge_index_math,
                   'chem': edge_index_chem}
edge_type_dict = {'phy': edge_type_phy,
                   'math': edge_type_math,
                   'chem': edge_type_chem}

edge_index = edge_index_dict[subject]
edge_type = edge_type_dict[subject]


max_length = 200

batch_size = 64
num_epochs = 20

concept_num  = 50+1
concept_dim1 = 256
concept_dim2 = 128
concept_dim3 = 64
num_relations = 2
diff_num     = 5+1
diff_dim     = 64
answer_num   = 2+1
answer_dim   = 64
output_dim1  = 64
output_dim2  = 64
hidden_dim   = 64
layer_dim    = 1

k1 = 0.01
k2 = 3.0
k_p = 0.001

lr = 0.001






