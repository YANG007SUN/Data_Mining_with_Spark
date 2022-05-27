from pyspark import SparkContext, SparkConf
import binascii
import random
import csv
from statistics import median
import sys
from time import time


start_time=time()
def create_hash_func_list(random_a,random_b):
    """generate num_hash number of hash functions
    """
    hash_list=[(a,b,largest_value) for a,b in zip(random_a,random_b)]
    return hash_list


def myhashs(s):
    """generate a num_hash number of hash value list for each s(id)
    """
    # create hash func list
    hash_func_list=create_hash_func_list(random_a,random_b)
    
    int_value=int(binascii.hexlify(s.encode("utf8")), 16)
    hash_values=[(params[0]*int_value+params[1])%params[2] for params in hash_func_list]
    return hash_values

class BlackBox:

    def ask(self, file, num):
        lines = open(file,'r').readlines()
        users = [0 for i in range(num)]
        for i in range(num):
            users[i] = lines[random.randint(0, len(lines) - 1)].rstrip("\n")
        return users

if __name__=='__main__':
    # params
    input_file=sys.argv[1]
    stream_size=int(sys.argv[2])
    num_of_asks=int(sys.argv[3])
    output_file=sys.argv[4]

    num_hash=100
    largest_value = sys.maxsize

    random_a=[random.randint(1,largest_value) for _ in range(num_hash)]
    random_b=[random.randint(1,largest_value) for _ in range(num_hash)]

    # main task
    bx=BlackBox()
    ct=0
    # 13 is the good chioce after testing
    num_groups=13
    group_size=int(num_hash/num_groups)
    final_results=[]


    for i in range(num_of_asks):
        final_numZero=[]
        stream_users=bx.ask(input_file,stream_size)
        # first value
        prev_hashValue_bit=list(map(lambda x: bin(x), myhashs(stream_users[0])))
        num_zero_prev=list(map(lambda x: len(str(x)) - len(str(x).rstrip("0")), prev_hashValue_bit))
        
        # find longest 0s for each batch
        for x in stream_users[1:]:
            new_hashValue_bit=list(map(lambda x: bin(x), myhashs(x)))
            num_zero_new=list(map(lambda x: len(x) - len(x.rstrip("0")), new_hashValue_bit))
            
            # get max 0s for each iteration
            final_numZero=[max(a,b) for a,b in zip(num_zero_prev,num_zero_new)]
            num_zero_prev=final_numZero
            
        # cal the median value and remove the highest and lowest
        sorted_value=sorted([2 ** i for i in final_numZero])
        
        # cut into groups
        grouped_value=[]
        for i in range(num_groups):
            grouped_value.append(sorted_value[group_size*i:group_size*(i+1)])
        
        final_list=[]
        for ls in grouped_value:
            final_list.append(sum(ls)/len(ls))
            
        ct+=round(median(final_list))
        final_results.append([i,stream_size,round(median(final_list))])

    
    # output
    with open(output_file,'w+') as f:
        writer=csv.writer(f)
        writer.writerow(['Time','Ground Truth','Estimation'])
        writer.writerows(final_results)

end_time=time()

print(f'Total time {end_time-start_time}')