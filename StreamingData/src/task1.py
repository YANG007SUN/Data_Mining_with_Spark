from pyspark import SparkContext, SparkConf
import binascii
import math
import random
import csv
import sys


def create_hash_func_list(n,random_a,random_b):
    """generate k number of hash functions
    """
    hash_list=[(a,b,n) for a,b in zip(random_a,random_b)]
    return hash_list


def myhashs(s):
    """generate a k length of hash value list for each s(id)
    """
    # create hash func list
    hash_func_list=create_hash_func_list(n,random_a,random_b)
    
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

    n=69997
    bit_array=[0 for _ in range(n)]

    # optimal k
    m=num_of_asks*stream_size
    k=round(n/m*math.log(2))

    # set sc
    conf=SparkConf().\
        set("spark.executor.memory", "4g").\
        set("spark.driver.memory", "4g")
    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR") 

    # generate list a and b for k hash funtions
    random_a=[random.randint(1,100) for _ in range(k)]
    random_b=[random.randint(1,100) for _ in range(k)]

    # Main tasks
    final_results=[]
    bx=BlackBox()
    prev_id=[]

    for i in range(num_of_asks):
        stream_users=bx.ask(input_file,stream_size)
        fp,tn=0,0
        # append first element in the batch
        prev_id.append(stream_users[0])
        hash_list= myhashs(stream_users[0])
        # hash into bit array
        for h in hash_list:
            bit_array[h]=1

        for x in stream_users[1:]:
            hash_list= myhashs(x)
            check=[1 for h in hash_list if bit_array[h]==1]
            # seen 
            if sum(check)==k:
                if x not in prev_id:
                    fp+=1
            # not seen
            if sum(check)!=k:
                if x not in prev_id:
                    tn+=1
            prev_id.append(x)
        fpr=fp/(fp+tn)
    #     print(fpr)
        final_results.append([i,fpr])
    
    
    with open(output_file,'w+') as f:
        writer=csv.writer(f)
        writer.writerow(['Time','FPR'])
        writer.writerows(final_results)