from pyspark import SparkContext, SparkConf
from collections import defaultdict
import sys
from itertools import combinations
import csv
import time

start_time=time.time()
def create_init_index(list_of_uid,user_dict):
    """convert list of user id to initial index
    """
    return [user_dict[uid] for uid in list_of_uid]

def hash_function(number,m,a)->int:
    """create a hash value based on 
       (i*a+99) % m
       where m: len(user_id), a: random number
    """
    return (a * number + 99) % m
    
def create_MHL(initial_list,n,m)->list:
    """create a min hash list based on initial index list
       return a min hash list with len of n
       m=len(user_id)
    """
    res=[]
    # for each hash function find out the min value of each hash list from one hash function
    for i in range(n):
        min_value=sys.maxsize # initialize min value
        for number in initial_list:
            # same hash function within each n
            hash_value=hash_function(number,m,i)
            if hash_value<min_value:
                min_value=hash_value
        res.append(min_value)
    return res

def create_LSH(bid,signature_m,bands,rows):
    """create LSH with n bands and each band with n rows
    """
    candidate_pairs=[]
    for n in range(bands):
        start_idx=n * rows
        end_idx=start_idx+rows
        
        local_sig=signature_m[start_idx:end_idx]
        local_sig.insert(0,n)
        hash_value=hash(tuple(local_sig))
        candidate_pair=(hash_value, bid)
        candidate_pairs.append(candidate_pair)
        
    return candidate_pairs

def create_candidate_pairs(list_of_list):
    """create candidate pairs after LSH
    """
    candidates=set()
    for ls in list_of_list:
        pairs=combinations(list(sorted(ls)),2)
        for pair in pairs:
            candidates.add(tuple(pair))
    return sorted(candidates)

def jaccard_similarity(users1, users2):
    s1=set(users1)
    s2=set(users2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def write_output(candidates,bid_map,output_path):
    res=[]
    for pair in candidates:
        b1,b2=pair[0],pair[1]
        users1,users2=bid_map[b1],bid_map[b2]
        sim=jaccard_similarity(users1,users2)
        if sim>=0.5:
            res.append([b1,b2,sim])
    with open(output_path,'w+') as f:
        writer=csv.writer(f)
        writer.writerow(['business_id_1','business_id_2','similarity'])
        writer.writerows(res)

if __name__=='__main__':
    input_path=sys.argv[1]
    output_path=sys.argv[2]
    bands=50
    rows=2

    # create spark session
    conf=SparkConf().\
     set("spark.executor.memory", "4g").\
     set("spark.driver.memory", "4g")
    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR") 
    
    # read data
    rdd=sc.textFile(input_path)
    header=rdd.first()
    new_rdd=rdd.filter(lambda r:r!=header).map(lambda r:r.split(',')[0:2])

    # create some variables
    bid_map=new_rdd.map(lambda r: (r[1], r[0])).\
            groupByKey().map(lambda r: (r[0], set(r[1]))).collectAsMap()

    user_id=rdd.filter(lambda r: r!=header).map(lambda r: r.split(',')[0]).distinct().collect()
    # business_id=rdd.filter(lambda r: r!=header).map(lambda r: r.split(',')[1]).distinct().collect()

    # create user index dict {user_id: index}
    user_dict={}
    for idx, uid in enumerate(user_id):
        user_dict[uid]=idx

    # signature matrix
    signature_m=new_rdd.map(lambda r:[r[1],r[0]]).\
            groupByKey().map(lambda r:[r[0],list(set(r[1]))]).\
            map(lambda r:[r[0],create_init_index(r[1],user_dict)]).\
            map(lambda r:[r[0],create_MHL(r[1],100,len(user_id))])

    # create LSH, business id with same hash value in same list
    hashed_list=signature_m.flatMap(lambda r: create_LSH(r[0],r[1],bands,rows)).\
            groupByKey().filter(lambda r: len(list(set(r[1])))>1).\
            flatMap(lambda r: [list(set(r[1]))]).collect()
    
    # create candidate pairs
    candidates=create_candidate_pairs(hashed_list)

    # output
    write_output(candidates,bid_map,output_path)

    end_time=time.time()
    print(f'total time {end_time-start_time}')



#     import pandas as pd
#     truth=pd.read_csv('./data/task1_task2.1/pure_jaccard_similarity.csv')
#     truth_vec=[r[1][0]+'-'+r[1][1]+'-'+str(r[1][2]) for r in truth.iterrows()]

#     output=pd.read_csv('./test.csv')
#     output_vec=[r[1][0]+'-'+r[1][1]+'-'+str(r[1][2]) for r in output.iterrows()]

#     precision=len(set(output_vec).intersection(output_vec))/len(output_vec)
#     recall=len(set(output_vec).intersection(output_vec))/len(truth_vec)
#     print(f'''
# Precision:{precision}
# Recall:{recall}
#     ''')