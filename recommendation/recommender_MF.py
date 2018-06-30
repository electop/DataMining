'''
Recommendation using Matrix Factorization
@Jae Kyun Kim
'''
import sys
import pandas as pd
import numpy as np

'''
Load input data
- Return user, item, rank with integer type
'''
def load_data(argument):
    user = list()
    item = list()
    rank = list()
    
    with open(argument, 'r') as f:
        input_data = f.read().split('\n') 
        input_data = input_data[:-1]
        for data in input_data:
            data = data.split('\t')
            ## User / Item / Rank / Timestamp
            user.append(data[0])
            item.append(data[1])
            rank.append(data[2])

    ## Convert string to integer
    user = list(map(lambda x: int(x), user))
    item = list(map(lambda x: int(x), item))
    rank = list(map(lambda x: int(x), rank))

    return user, item, rank

'''
1. Make matrix size based on max length of user, item 
(Better than array since only use 2-dimensonal)
2. Insert rank data
'''
def make_matrix(user, item, rank):
    matrix = np.zeros(shape = (max(user), max(item)))

    ## Insert rank data    
    k = 0
    loc = np.column_stack((user, item))
    for i in range(len(loc)):
        matrix[loc[i][0] - 1, loc[i][1] - 1] = rank[k]
        k += 1

    return matrix 

'''
Matrix factorization
'''
def matrix_factorization(R, P, Q, K):
    learning_rate = 5000
    ## Rate of approaching the minimum
    alpha = 0.0002
    beta = 0.02

    print('Optimization Start..!')
    for count in range(learning_rate):
        print('Learning Step:', count + 1)
        for i in range(len(R)):
            for j in range(len(R[0])):
                ## if there is rate score 
                if R[i][j] > 0:
                    ## Error = Rate - estimated rate
                    error = R[i][j] - np.dot(P[i,:], Q[:,j])
                    ## Minimizing the error using gradient method
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * error * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * error * P[i][k] - beta * Q[k][j])
        
        for i in range(len(R)):
            for j in range(len(R[0])):
                R[i][j] = np.dot(P[i,:], Q[:,j])
                ## Adjust the value from 1~5 rating score 
                if R[i][j] > 5:
                    R[i][j] = 5
                if R[i][j] < 1:    
                    R[i][j] = 1
    
    print('Learning Finished')
    return R


if __name__ == '__main__':
    ## Load data and make matrix
    user, item, rank = load_data(sys.argv[1])
    test_user, test_item, test_rank = load_data(sys.argv[2])    
    
    ## Make matrix 
    R = make_matrix(user, item, rank)  

    ## N: matrix row, M: matrix col 
    N = len(R)
    M = len(R[0])
    ## Bigger, more accurate
    K = 2
    ## Make random P, Q matrix
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)
    Q = Q.T

    ## Matrix Factorization
    R = matrix_factorization(R, P, Q, K)
    
    print(R)

    

