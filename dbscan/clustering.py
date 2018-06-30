'''
Clustering using DBSCAN
@Jae Kyun Kim
'''

import sys
import math

## Global Variable:: Data information about cluster label type
cluster_labels = list()

'''
Load input data and return in list
Data format: [ID, x_cor, y_cor]
'''
def load_data(argument):
    df = list()
    
    with open(argument, 'r') as f:
        input_data = f.read().split('\n') 
        for data in input_data:
            data = data.split('\t')
            df.append(data)
    
    return df[:-1]

'''
Check if the data is neighbor::
If the distance between ID and compare ID less then eps, it is neighbor 
'''
def is_neighbor(cor_1, cor_2, eps):
    ## Converting string to float 
    x1 = float(cor_1[0])
    y1 = float(cor_1[1])
    x2 = float(cor_2[0])
    y2 = float(cor_2[1])

    distance = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    
    ## True if it is neighbor
    return distance < eps

'''
Search neighbor based on input ID using eps
'''
def search_neighbor(total_data, ID, eps):
    neighbor_list = list()
    ## Only x_cor, y_cor is needed
    cor_1 = total_data[ID][1:]
    
    ## Search all data if it is neighbor or not 
    for neighbor_id in range(len(total_data)):
        cor_2 = total_data[neighbor_id][1:]
        if is_neighbor(cor_1, cor_2, eps):
            ## If data is neighbor then append it to the neighbor list
            neighbor_list.append(neighbor_id)
    
    return neighbor_list

'''
Classify Clustering Process
1. Search neighbors based on input ID
2. Expand the cluster by DFS method with neighbors 
'''
def classify_cluster(total_data, ID, label_type, eps, minPts):
    global cluster_labels
    
    ## Search neighbors
    neighbors = search_neighbor(total_data, ID, eps)

    ## Neighbor more than minPts -> Make cluster
    if len(neighbors) >= minPts:
        cluster_labels[ID] = label_type
        
        ## Give same cluster label type to neighbors
        for neighbor_id in neighbors:
            cluster_labels[neighbor_id] = label_type

        ## Loop all neighbors until there is no more core point        
        while len(neighbors) > 0:
            neighbor_id = neighbors[0]
            ## Recursively expand cluster based on DFS method
            ## Search child neighbor
            child_list = search_neighbor(total_data, neighbor_id, eps)
            
            ## if child neighbor's count is more than minPts, it is core point
            if len(child_list) >= minPts:
                    
                for i in range(len(child_list)):
                    index = child_list[i]
                    ## If child's neighbor not visited or is outlier
                    if cluster_labels[index] == None or cluster_labels[index] == -1:
                        
                        ## Append it to neighbor list
                        neighbors.append(index)
                        cluster_labels[index] = label_type
            
            ## Go to next         
            neighbors = neighbors[1:]
        
        return True
    
    ## Neighbor less than minPts
    else:
        cluster_labels[ID] = -1
        return False

'''
Optimization is optional::
Not applicating in this case
'''
# def optimization(cluster_number):
#     cluster_label_count = len(set(cluster_labels))
#     print(cluster_label_count)
#     ## If cluster label count is more than givien cluster number
#     if cluster_label_count > cluster_number:
#         print("Pruning..!")
#         for index in range(len(cluster_labels)):
#             if cluster_labels[index] >= cluster_number:
#                 print(cluster_labels[index], cluster_number)
#                 cluster_labels[index] = cluster_number - 1

'''
Write data from each cluster label
'''
def write_data(input_name):
    input_name = input_name.replace('.txt', '')
    
    for label in range(len(set(cluster_labels))-1):
        file_name = input_name + '_cluster_' + str(label) + '.txt'
        
        ## Pull out the index numbers based on cluster labels
        ID_list = [i for i, j in enumerate(cluster_labels) if j == label]
        
        temp = ''
        for id in ID_list:
            temp += str(id) + '\n'
        with open(file_name, 'w') as f:
            f.write(temp)


if __name__ == '__main__':

    ## Load input data 
    total_data = load_data(sys.argv[1])
    cluster_number = int(sys.argv[2])
    eps = int(sys.argv[3]) ## epsilon
    minPts = int(sys.argv[4])

    ## Cluster label starting from 0
    label_type = 0
    ## Initializing cluster labels with None
    cluster_labels = [None] * len(total_data)

    print("DBSCAN START:: \n")

    ## Loop every id in total data
    for ID in range(len(total_data)):
        ## If ID not visited then make cluster
        if cluster_labels[ID] == None:
            if classify_cluster(total_data, ID, label_type, eps, minPts):
                print("Cluster Label Type %s:: Classified..!" % label_type)
                label_type = label_type + 1
    
    ## Optimize cluster with given cluster number
    # optimization(cluster_number)

    print("Finished.")

    ## Write File
    write_data(sys.argv[1])


