import sys
from pyspark import SparkContext
import json


if __name__=='__main__':
    # get file path and output file name
    file_path=sys.argv[1]
    output_path=sys.argv[2]

    # create spark context
    sc=SparkContext.getOrCreate()
    sc.setLogLevel("ERROR") 

    # read in data
    rdd_review=sc.textFile(file_path).map(lambda r:json.loads(r))

    results={}
    # A
    results['n_review']=rdd_review.count()

    # B
    results['n_review_2018']=rdd_review.\
        map(lambda r:r['date'][:4]).\
        filter(lambda y:int(y)==2018).count()

    # C
    results['n_user'] = rdd_review.\
        map(lambda r:(r['user_id'],1)).\
        distinct().count()

    # D
    results['top10_user']=rdd_review.map(lambda r:(r['user_id'],r['review_id'])).\
        groupByKey().\
        map(lambda r:[r[0],len(list(r[1]))]).\
        takeOrdered(10,key=lambda r:(-r[1], r[0]))

    # E
    results['n_business'] = rdd_review.\
        map(lambda r:(r['business_id'],1)).\
        distinct().count()

    # F
    results['top10_business']=rdd_review. \
        map(lambda r:(r['business_id'],r['review_id'])).\
        groupByKey().\
        map(lambda r:[str(r[0]),len(list(r[1]))]).\
        takeOrdered(10,key=lambda r:(-r[1], r[0]))

    # # output results
    # results={
    #     'n_review':n_review,
    #     'n_review_2018':n_review_2018,
    #     'n_user':n_users,
    #     'top10_user':top10_users,
    #     'n_business':n_business,
    #     'top10_business':top10_business   
    # }
    
    with open(output_path,"w+") as output:
        json.dump(results,output)

    output.close()

    print(results)