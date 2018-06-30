'''
Decision Tree Algorithm
@Jae Kyun Kim
'''

import pandas as pd
import math
import sys

'''
Tree를 Class로 구현
- Node는 attribute를 들고 있고, Node의 Children이 dictionary 형태로 저장:: Key(attribute의 label): Value(다음 attribute)
- Top-down recursive divide and conquer
'''
class Node():
    ## Build Tree recursively 
    ## Default node_type is attribute
    def __init__(self, dataframe, node_type='attribute'):
        self.node_type = node_type
        self.dataframe = dataframe
        
        ## Save decision at node level in case test data can't get to bottom level
        ## Choose based on high frequency decision
        self.decision = self.dataframe.iloc[:,-1].value_counts().idxmax()
        
        ## if the node_type is attribute level then build the tree
        if node_type == 'attribute':
            self.build_tree()
        ## Leaf:: if the node_type is decision level then get Class Label 
        elif node_type == 'decision':
            self.decision = self.dataframe.iloc[:,-1].value_counts().idxmax()

    ## Check if we can still build the tree        
    def no_more_build(self, dataframe):
        ## Is all decision unanimous ?
        decision = dataframe.iloc[:,-1].unique()
        if len(decision) == 1:            
            self.no_need_classify = True
        ## Is there any attribute left ? 
        ## I made tree depth not too deep making 2 -> 3
        elif len(dataframe.columns) == 3:
            self.no_need_classify = True
        ## If not keep build!
        else:
            self.no_need_classify = False

        return self.no_need_classify
    
    ## Build Tree based on Gain Ratio
    def build_tree(self):
        ## Find next attribute using Gain Ratio
        self.attribute = get_max_gain_ratio(self.dataframe)
        ## Send selected attribute column to classify data by label
        self.classify_by_label(self.dataframe[self.attribute])

    ## Classify data by label    
    def classify_by_label(self, column):
        labels = column.unique()
        ## Make children from node by dictionary
        self.children = dict()

        ## Make another Tree node by labels
        for label in labels:
            ## Send sorted dataframe for next Node
            sorted_dataframe = self.dataframe.loc[self.dataframe[self.attribute] == label]
            if self.no_more_build(sorted_dataframe):
                ## giving dataframe after sorting with label
                self.children[label] = Node(sorted_dataframe, node_type='decision')
            else:
                ## giving dataframe after deleting attribute column
                self.children[label] = Node(sorted_dataframe.drop(self.attribute, 1))

    ## TEST the new data:: line by line (Series)
    def search(self, series):
        ## if it is leaf then get the final decision
        if self.is_leaf():
            return self.decision
        
        else:
            ## Search from root node
            label = series[self.attribute]
            try:
                child_node = self.children[label]
            except:
                ## if there is no label then just get the Node Decision
                return self.decision
                ## Recursively searching
            return child_node.search(series)

    ## Check if it is leaf node
    def is_leaf(self):
        return self.node_type == 'decision'

## Expected information(entropy) needed to classify a tuple in D
def get_info_entropy(values):
    total = values.sum()
    summation = 0
    for value in values:
        if value:
            summation += value / total * math.log(value/total) / math.log(2)
        else:
            return 0
    return -summation

## Get gain ratio after using A to split D into V partitions
def get_gain_ratio(table):
    total = table.values.sum()
    gain_sum = 0
    ## Gain
    for row in table.values:
        gain_sum += row.sum() / total * get_info_entropy(row)

    ## SplitInfo
    split_sum = 0
    for row in table.values:
        x = row.sum() / total
        split_sum += x * math.log(x) / math.log(2)

    gain_ratio = gain_sum / -split_sum
    return gain_ratio

## return max gain ratio among the attributes
def get_max_gain_ratio(dataframe):
    candidates = dict()
    ## making class label in col 
    decision_class = dataframe[dataframe.columns[-1]]
    for column in dataframe.columns[:-1]:
        candidate = dataframe[column]
        table = pd.crosstab(candidate, decision_class)
        candidates[column] = get_gain_ratio(table)

    ## Give minimum gain_ratio which is maximum gain ratio 
    ## gain = info(D) - info_a(D)    
    return min(candidates.keys(), key=(lambda k: candidates[k]))

def load_data(argument):
    df = list()
    with open(argument, 'r') as f:
        input_data = f.read().split('\n') 
        for data in input_data:
            data = data.split('\t')
            df.append(data)
    
    ## converting list of lists into pandas DataFrame
    df = pd.DataFrame(df)
    ## Replacing header with top row
    header = df.iloc[0]
    df = df[1:-1]
    df.columns = header
    return df

## sys.argv[3]:: output.txt
def write_data(line):
    with open(sys.argv[3], 'a') as f:
        new_line = ''
        for word in line:
            new_line += word+'\t' 
        f.write(new_line[:-1] + '\n') 


if __name__ == '__main__':
    ## load train, test data and store to pd.df
    df_train = load_data(sys.argv[1])
    df_test = load_data(sys.argv[2])

    ## Make Decision Tree root node
    tree = Node(df_train)

    ## Write result to output.txt
    header = df_train.columns
    header = list(header)
    write_data(header)

    for _, row in df_test.iterrows():
        line = row.values
        result = tree.search(row)
        line = list(line)
        line.append(result)

        write_data(line)        
        

