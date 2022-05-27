from pyspark import SparkContext, SparkConf
import random
import csv
import sys

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

    s=100

    reservoir=[]
    i_pointer=0
    final_results=[]

    bx=BlackBox()
    random.seed(553)
    for i in range(num_of_asks):
        # call each batch    
        stream_users=bx.ask(input_file,stream_size)
        for x in stream_users:
            
            i_pointer+=1
            # reservior still has space
            if len(reservoir)<s:
                reservoir.append(x)
            # reservior is full
            else:
                prob=stream_size/i_pointer
                if random.random()<prob:
                    random_idx=random.randint(0,stream_size-1)
                    reservoir[random_idx]=x
                    
        if len(reservoir)>=s:
            final_results.append([i,reservoir[0],reservoir[20],reservoir[40],reservoir[60],reservoir[80]])
    
    with open(output_file,'w+') as f:
        writer=csv.writer(f)
        writer.writerow(['seqnum','0_id','20_id','40_id','60_id','80_id'])
        writer.writerows(final_results)