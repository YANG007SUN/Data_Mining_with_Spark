from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict
import sys
import time
import csv


start_time=time.time()
def create_L1_phase1(basket_partition,support)->list:
    """create frequent singletons for SON Phase1,return a list
    """
    frequent_items=[]
    d=defaultdict(int)

    # loop through bask to find count of singletons
    for b in basket_partition:
        for item in b:
            d[item]+=1
    # create L1
    L_1=[k for k,v in d.items() if v>=support]
    frequent_items.append(L_1) # frequent singletons
    
    return frequent_items,L_1

def create_C_k(basket,freq_singletons,k)->dict:
    """create candidates itemsets dictionary for k>=2
    """
    d=defaultdict(int)
    for b in basket:
        # only create combo for subset of each basket
        new_b=b
        for item in b:
            if item not in freq_singletons:
                new_b.remove(item)
        # sorted list due to order of combo will create a different key
        combos=combinations(sorted(new_b),k)
        for item in combos:
            # list is unhashble
            d[tuple(item)]+=1
    return d

def update_singletons(L_k):
    """Given list of L_k, re-construct L_1.
       Idea is if certain singletons not showing up in prev sets, it will not show up anymore.
    """
    L_1={}
    for l in L_k:
        L_1=set(l).union(L_1)
    return list(L_1)

def create_frequent_set_phase_1(basket_length,basket_partition,support:int)->list:
    """create the frequent itemset for SON algorithm phase 1
    """
    
    # get the new paritition support value
    basket_partition=list(basket_partition)
    p_size,b_size=len(list(basket_partition)),basket_length
    new_support=int((p_size/b_size)*support)+1
    
    # create L_1 and freq set for final output
    frequent_items,L_1=create_L1_phase1(basket_partition,new_support)

    # initital k=2
    k=2
    while 1:
        C_k=create_C_k(basket_partition,L_1,k)
        L_k=[k for k,v in C_k.items() if v>=new_support]
        if len(L_k)==0:
            return frequent_items
        frequent_items.append(L_k)
        L_1=update_singletons(L_k)
        k+=1
        
    return frequent_items   

def create_frequent_set_phase_2(basket:list,phase1_itemsets)->list:
    """create the frequent itemset for SON algorithm phase 2
    """
    frequent_items=defaultdict(int)
    basket=list(basket)
    for b in basket:
        for item in phase1_itemsets:
            if set(item).issubset(set(b)):
                frequent_items[tuple(item)]+=1
    
    return frequent_items.items()

def format_list(freq_sets):
    """format list of freq set into specified format
    """
    res=''
    for i,k in enumerate(freq_sets):
        if len(k)==1:
            new_k=str(k).replace(',','')
        else:
            new_k=str(k)
        # dont go out of len
        if i<len(freq_sets)-1:
            # check next len is greater than current
            if len(k)<len(freq_sets[i+1]):
                res+=f'{new_k}\n\n'
            else:
                res+=f'{new_k},'
        else:
            res+=f'{new_k}'
    return res

def write_output(format_phase_1:list,format_phase_2:list,output_file):
    """output results with specified format
    """
    with open(output_file,'w+') as res:
        res.write(f'Candidates:\n{format_phase_1}\n\nFrequent Itemsets:\n{format_phase_2}')


def PreProcessFile(input_path,output_raw_path):
    results=[]
    with open(input_path,'r') as f:
        csvreader=csv.reader(f)
        # skip header pointer
        header=next(csvreader)
        for r in csvreader:
            results.append([str(r[0]) +'-'+ str(r[1]), str(int(r[5]))])
        # write out
        with open(output_raw_path,"w+") as f:
            writer = csv.writer(f)
            writer.writerows(results)


if __name__=='__main__':
    filter_threshold=int(sys.argv[1])
    support=int(sys.argv[2])
    input_path_raw=sys.argv[3]
    output_path=sys.argv[4]
 
    # create spark session
    sc=SparkContext.getOrCreate()
    sc.setLogLevel("ERROR") 

    # process the file and transform to rdd
    PreProcessFile(input_path_raw,'processed.csv')

    rdd=sc.textFile('processed.csv')

    basket=rdd.map(lambda r: (r.split(',')[0], r.split(',')[1])).\
               groupByKey().\
               map(lambda r:list(set(r[1]))).\
               filter(lambda r:len(r)>filter_threshold)

    basket_length=basket.count()
    
    # SON phase1
    phase_1=basket.mapPartitions(lambda p:create_frequent_set_phase_1(basket_length,p,support)).\
                   flatMap(lambda r:r).\
                   distinct().\
                   map(lambda r: r if type(r) is tuple else (r,)).\
                   sortBy(lambda r:(len(r),r)).collect()
    # SON phase2
    phase_2=basket.mapPartitions(lambda p:create_frequent_set_phase_2(p,phase_1)).\
                   map(lambda r:[r[0],r[1]]).\
                   groupByKey().\
                   mapValues(sum).\
                   filter(lambda r:r[1]>=support).collect()
    freq_sets=sorted([i[0] for i in phase_2],key=lambda r:(len(r),r))

    # write output
    write_output(format_list(phase_1),format_list(freq_sets),output_path)

    end_time=time.time()
    print(f'Duration:{end_time-start_time}')
