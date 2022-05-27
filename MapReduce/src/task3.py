from pyspark import SparkContext
import json
import sys
import time


if __name__=='__main__':

    # get data
    review_path=sys.argv[1]
    business_path=sys.argv[2]
    output_path_a=sys.argv[3]
    output_path_b=sys.argv[4]

    # spark session
    sc=SparkContext.getOrCreate()
    sc.setLogLevel("ERROR") 
    
    # read in data to RDD
    review_rdd=sc.textFile(review_path).map(lambda r:json.loads(r))
    business_rdd=sc.textFile(business_path).map(lambda b:json.loads(b))

    # ========================================== Task A ==========================================
    review=review_rdd.\
        map(lambda b:[b['business_id'],b['stars']])
    business=business_rdd.\
        map(lambda b:[b['business_id'],b['city']])

    results_A=business.join(review).\
        map(lambda b:list(b[1])).\
        groupByKey().\
        mapValues(lambda b: sum(b) / len(b)) .\
        sortBy(lambda b: (-b[1],b[0])).collect()
    
    # output json file
    output_A = ['city','stars\n']
    for l in results_A:
        output_A.append(str(l[0])+ ',' + str(l[1]) + '\n')

    with open(output_path_a,"w+") as output1:
        json.dump(output_A,output1)
    output1.close()
    

    # ========================================== Task B ==========================================
    # python sort
    start=time.time()
    unsorted_A=business.join(review).\
                map(lambda b:list(b[1])).\
                groupByKey().\
                mapValues(lambda b: sum(b) / len(b)).collect()
    sorted(unsorted_A,key=lambda b:-b[1])
    end=time.time()
    m1=end-start

    # spark sort
    start=time.time()
    sorted_A=business.join(review).\
                map(lambda b:list(b[1])).\
                groupByKey().\
                mapValues(lambda b: sum(b) / len(b)).\
                sortBy(lambda b: (-b[1],b[0])).collect()
    end=time.time()
    m2=end-start

    result_B={
        'm1':m1,
        'm2':m2,
        'reason':'python sort is faster than spark sort because python sort does not need to shuffle and aggregate data from map task. Also, spark sort has two steps (transformation and action), and the previous transformation is evaluated after calling an action, and it takes moer tiem for the steps. On the other hand, python loads the data at once and evaluated the data at one time. '
    }

    # output task B file
    with open(output_path_b,'w+') as output2:
        json.dump(result_B,output2)
    output2.close()
    