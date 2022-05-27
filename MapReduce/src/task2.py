import sys
from pyspark import SparkContext
import json
import time


if __name__=='__main__':

    final_output={}

    # get data
    file_path=sys.argv[1]
    output_path=sys.argv[2]
    n_partition=int(sys.argv[3])

    # create sc session
    sc=SparkContext.getOrCreate()
    sc.setLogLevel("ERROR") 
    
    # read data
    rdd_review = sc.textFile(file_path).map(lambda r:json.loads(r))

    # task1 QF
    result_F = rdd_review.map(lambda r:(r['business_id'],1))

    # ============================  Default partition  ============================
    default={}
    default_partition=result_F.getNumPartitions()

    # n_items
    n_items_default=result_F.glom().map(len).collect()

    # average time
    start_defauld=time.time()
    rdd_review=sc.textFile(file_path).map(lambda r:json.loads(r))
    F=rdd_review. \
        map(lambda r:(r['business_id'],r['review_id'])).\
        groupByKey().\
        map(lambda r:[str(r[0]),len(list(r[1]))]).\
        takeOrdered(10,key=lambda r:(-r[1], r[0]))
    end_default=time.time()
    exe_time_default=end_default-start_defauld

    # result
    default['n_partition']=default_partition
    default['n_items']=n_items_default
    default['exe_time']=exe_time_default


    # ============================  cust partition  ============================
    cust={}

    # n_items
    n_items_cust=result_F.partitionBy(n_partition).glom().map(len).collect()

    # average time
    start_cust=time.time()
    rdd_review=sc.textFile(file_path).map(lambda r:json.loads(r))
    F=rdd_review. \
        map(lambda r:(r['business_id'],r['review_id'])).\
        partitionBy(n_partition).\
        groupByKey().\
        map(lambda r:[str(r[0]),len(list(r[1]))]).\
        takeOrdered(10,key=lambda r:(-r[1], r[0]))
    end_cust=time.time()
    exe_time_cust=end_cust-start_cust

    # result
    cust['n_partition']=n_partition
    cust['n_items']=n_items_cust
    cust['exe_time']=exe_time_cust

    final_output['default']=default
    final_output['customized']=cust

    # write out file
    with open(output_path,"w+") as output:
        json.dump(final_output,output)

    output.close()

    print(final_output)