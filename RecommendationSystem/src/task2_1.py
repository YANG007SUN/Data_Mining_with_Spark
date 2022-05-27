from pyspark import SparkContext, SparkConf
from collections import defaultdict
import sys
import csv
import math
import time

start_time=time.time()
def create_dict(ls_of_dict):
    """create business user dictionary with format
       {business:{user:{rating}}}
    """
    new_dict=defaultdict(dict)
    for row in ls_of_dict:
        for ur in row[1]:
            new_dict[row[0]][ur[0]]=ur[1]
    return new_dict

def calculate_pearson_sim(b1,b2,co_rated_number):
    """calculate pearson sim based on two bids
       b1: new bid
       b2: bid in training data
       reutrned sim is all greater than 0. or 0
    """
    u1=business_dict[b1]
    u2=business_dict[b2]
        
    # if new bid never exist in training data, sim=0
    if len(u1)==0:
        return 0
        
    ur1=set(u1.keys())
    ur2=set(u2.keys())
    co_rated_users=ur1&ur2
        
    if len(co_rated_users)>=co_rated_number:
        ur1_rating=[float(u1[i]) for i in co_rated_users]
        ur2_rating=[float(u2[i]) for i in co_rated_users]

        avg_ur1=sum(ur1_rating)/len(ur1_rating)
        avg_ur2=sum(ur2_rating)/len(ur1_rating)

        numerator=sum((u-avg_ur1)*(v-avg_ur2) for u,v in zip(ur1_rating,ur2_rating))
        denominator=math.sqrt(sum((u-avg_ur1)**2 for u in ur1_rating))*\
                    math.sqrt(sum((v-avg_ur2)**2 for v in ur2_rating))

        if numerator>0 and denominator>0:
            return float(numerator/denominator)
        else:
            return 0
    return 0


def create_prediction_params(new_bid,new_uid):
    """create prediction params
       return a list [(new_bid,current_bid),(sim_for_two_bid,current_bid_rating)] for all related current bid
    """
    pearson_rating_list=[]
    # get list of bid that new uid has rated and its ratings
    bids=list(user_dict.get(new_uid,[]).keys())
    ratings=list(user_dict.get(new_uid,[]).values())

    for current_bid,current_rating in zip(bids,ratings):
        sim=calculate_pearson_sim(new_bid,current_bid,co_rated_number)
        # record only if sim!=0
        if sim!=0:
            pearson_rating_list.append([sim,float(current_rating)])
    return pearson_rating_list

def make_predition(pearson_rating_list,bid):
    """create predition
    """
    numerator=sum(r[0]*r[1] for r in pearson_rating_list)
    denominator=sum(abs(r[0]) for r in pearson_rating_list)
    if numerator==0 or denominator==0:
        return business_avg_dict[bid]
    prediction=float(numerator/denominator)
    return prediction

def write_output(prediction_results,output_file):
    """write csv output
    """
    with open(output_file,'w+') as f:
        writer=csv.writer(f)
        writer.writerow(['user_id','business_id','prediction'])
        writer.writerows(prediction_results)

if __name__=='__main__':
    # file path
    training_file=sys.argv[1]
    testing_file=sys.argv[2]
    output_file=sys.argv[3]

    # define params
    co_rated_number=30


    # create spark sc
    conf=SparkConf().\
     set("spark.executor.memory", "4g").\
     set("spark.driver.memory", "4g")
    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR") 

    # read in training and validation data
    rdd_train=sc.textFile(training_file)
    rdd_val=sc.textFile(testing_file)
    header=rdd_train.first()
    rdd_train=rdd_train.filter(lambda r:r!=header)
    rdd_val=rdd_val.filter(lambda r:r!=header)

    # create a business dictionay: {business:{user:{rating}}}
    business_rdd=rdd_train.\
                map(lambda r:r.split(',')).\
                map(lambda r: (r[1], (r[0], r[2]))).\
                groupByKey().\
                map(lambda x: (x[0], list(x[1]))).collect()

    # create a users dictionay: {user:{business:{rating}}}
    user_rdd=rdd_train.\
                map(lambda r:r.split(',')).\
                map(lambda r: (r[0], (r[1], r[2]))).\
                groupByKey().\
                map(lambda x: (x[0], list(x[1]))).collect()

    # avarege rating by business
    business_avg=rdd_train.map(lambda r:(r.split(',')[1],float(r.split(',')[2]))).\
                groupByKey().\
                mapValues(lambda x: sum(x) / len(x)).collect()
    business_dict=create_dict(business_rdd)
    user_dict=create_dict(user_rdd)
    # create business average rating dictionary
    business_avg_dict=defaultdict(float)
    for row in business_avg:
        business_avg_dict[row[0]]=row[1]

    # create predictions
    prediction_results=rdd_val.map(lambda r:r.split(',')[0:2]).\
                    map(lambda r:[r[0],r[1],make_predition(create_prediction_params(r[1],r[0]),r[1])]).\
                    collect()

    # output prediction files
    write_output(prediction_results,output_file)

    end_time=time.time()

    print(f'Total time: {end_time-start_time}')
