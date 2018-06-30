'''
Recommendation using Collaborative Filtering
@Jae Kyun Kim 
'''

import math
import sys

'''
Pearson correlation
- Most popular correlation among pearson, spearman etc
'''
def p_correlation(trains, new_user, train_user):
    ## Compare the common items between new user and train user  
    common_items = dict()
    for item in trains[new_user]:
        if item in trains[train_user]: 
            common_items[item] = 1
    
    count = len(common_items)
    ## If there is at least one common item, get correlation coefficient
    if count != 0: 
        new_sum, train_sum, new_sum_2, train_sum_2, product_sum = 0, 0, 0, 0, 0
        
        ## Sums of ratings 
        for item in common_items:
            new_sum += trains[new_user][item]
        for item in common_items:
            train_sum += trains[train_user][item]

        ## Sums of squares of ratings
        for item in common_items:
            new_sum_2 += trains[new_user][item] * trains[new_user][item] 
        for item in common_items:
            train_sum_2 += trains[train_user][item] * trains[train_user][item]

        ## Sum of the products
        for item in common_items:
            product_sum += trains[new_user][item] * trains[train_user][item] 
        
        ## Get correlation coefficient 
        a = product_sum - (new_sum * train_sum / count)
        b = math.sqrt((new_sum_2 - (new_sum * new_sum) / count)\
         * (train_sum_2 - (train_sum * train_sum) / count))
        
        ## Denominator cannot be 0
        if b == 0:
            return 0

        return a / b

    ## if there is no common item, then no correlation        
    else:
        return 0

'''
Get recommendations from train set based on new user
'''
def user_recommendation(trains, new_user):
    item_coefficient = dict()
    item_rating = dict()
    total_set = list()

    ## Get every correlation with train user
    for train_user in trains:
        if train_user != new_user:
            ## Pearson correlation
            coefficient = p_correlation(trains, new_user, train_user)
            ## If both users have correlation
            if coefficient > 0:
                for item in trains[train_user]:
                    ## Find the unrated item
                    if item not in trains[new_user] or trains[new_user][item] == 0:
                        item_coefficient.setdefault(item, 0)
                        item_coefficient[item] += coefficient
                        
                        item_rating.setdefault(item, 0)
                        ## Multiply coefficient to train user rating
                        item_rating[item] += trains[train_user][item] * coefficient

    coef_set = [(total / item_coefficient[item], item) for item, total in item_rating.items()]

    return coef_set

'''
CF algorithm 
'''
def collaborative_filtering(trains, tests):
    total_user = list()
    new = set()
    get_rate = list()
    
    ## Loop each users in test set 
    for line in tests:
        test_user = line[0]
        test_item = line[1]
        
        ## If user have not been recommended then get user recommendation
        if test_user not in new:
            coef_set = user_recommendation(trains, test_user)

        ## Use set to avoid repeated users
        new.add(test_user)

        ## Save rate
        for item in coef_set:
            if item[1] == test_item:
                get_rate.append(item[0])

        ## Compare the test item in recommendation set
        if len(get_rate) == 0: 
            get_rate = [5]
        
        line = [test_user, test_item, get_rate[0]]
        total_user.append(line)

    return total_user


'''
Load train data and return in dictionary
Load test data and return in list 
- trains:: {User: {item1: rating, item2: rating}}
- tests:: [(user, item, rating, ts), (user, item, rating, ts)] 
'''
def load_data(file_name, file_name2):
    trains = {}
    tests = []

    ## Open train data
    with open(file_name, 'r') as f:
        input_data = f.read().split('\n')
        input_data = input_data[:-1]

        for line in input_data:
            data = line.split('\t')
            user, item, rating, ts = data[0], data[1], data[2], data[3]
            ## Making dictionary
            ## {User: {item1: rating, item2: rating}}
            trains.setdefault(user, {})
            trains[user][item] = int(rating)

    ## Open test data
    with open(file_name2, 'r') as f:
        input_data = f.read().split('\n')
        input_data = input_data[:-1]

        for line in input_data:
            data = line.split('\t')
            user, item, rating, ts = data[0], data[1], data[2], data[3]
            tests.append((user, item, rating, ts))

    return trains, tests 

'''
Write data
'''
def write_data(input_name, recommended_set):
    output_name = input_name + '_prediction.txt'
    with open(output_name, 'w') as f:
        for i in range(len(recommended_set)):
            line = ''
            for j in range(len(recommended_set[0])):
                line += '%s\t' % (recommended_set[i][j])
            line = line[:-1]
            line += '\n'
            
            f.write(line)

if __name__ == "__main__":
    ## Load data and make trains, tests dataset 
    trains, tests = load_data(sys.argv[1], sys.argv[2])
    
    ## CF algorithm: return recommended set
    recommended_set = collaborative_filtering(trains, tests)

    ## Write data
    write_data(sys.argv[1], recommended_set)
    
