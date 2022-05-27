
from pyspark import SparkContext, SparkConf
from collections import defaultdict
import sys
import csv
import math
import json
import xgboost as xgb
from operator import add, mod
from time import time
import numpy as np

start_time=time()

# ======================================== Item based function ==========================================
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

# ======================================== Model based function ==========================================
def business_features(business_path):
    """create additional features from business.json file
    """
    business=sc.textFile(business_path).map(lambda r: json.loads(r)).\
            map(lambda r:[r['business_id'],(r['stars'],r['review_count'])])
    return business


def user_features(user_path):
    """create additional features from user.json file
    """
    user=sc.textFile(user_path).map(lambda r:json.loads(r)).\
            map(lambda r: [r['user_id'],(r['review_count'],r['average_stars'])])
    return user


def create_train_test_mat(rdd_train,rdd_test,business_features,user_features):
    """create X mat for training and testing data 
       return mat include both X and y
    """
    model_training=rdd_train.map(lambda r: r.split(',')).map(lambda r:[r[1],(r[0],float(r[2]))]).\
        join(business_features).\
        map(lambda r:[r[0],(r[1][0][0],r[1][0][1],r[1][1][0],r[1][1][1])]).\
        map(lambda r: [r[1][0],(r[0],r[1][1],r[1][2],r[1][3])]).\
        join(user_features).\
        map(lambda r:[r[0],r[1][0][0],r[1][0][1],r[1][0][2],r[1][0][3],r[1][1][0],r[1][1][1]])
    
    model_testing=rdd_test.map(lambda r: r.split(',')).map(lambda r:[r[1],(r[0])]).\
        join(business_features).\
        map(lambda r:[r[0],(r[1][0],r[1][1][0],r[1][1][1])]).\
        map(lambda r: [r[1][0],(r[0],r[1][1],r[1][2])]).\
        join(user_features).\
        map(lambda r:[r[0],r[1][0][0],r[1][0][1],r[1][0][2],r[1][1][0],r[1][1][1]])
    # create copy of testing data for outputing
    output_testing=model_testing.map(lambda r:r[0:2]).collect()
    return model_training,model_testing,output_testing

def create_X_y(model_training,model_testing):
    """create X_train,X_test,y_train
    """
    X_train=model_training.map(lambda r:r[3:]).collect()
    y_train=model_training.map(lambda r:r[2]).collect()
    X_test=model_testing.map(lambda r:r[2:]).collect()

    # make it an shaped array
    X_train_shape=np.array(X_train)
    y_train_shape=np.array(y_train)
    X_test_shape=np.array(X_test)
    
    return X_train_shape,y_train_shape,X_test_shape
    
def output_file(final_output,output_path):
    with open(output_path,'w+') as f:
        writer=csv.writer(f)
        writer.writerow(['user_id','business_id','prediction'])
        writer.writerows(final_output)

def model(X_train_shape,y_train_shape):
    model = xgb.XGBRegressor()
    model.fit(X=X_train_shape, y=y_train_shape)
    
    return model

def calculate_rmse(true_value, predictions):
    rmse_sum = sum([(true_value[i]-predictions[i])**2 for i in range(len(true_value))])
    rmse= math.sqrt(rmse_sum/len(true_value))
    return rmse

# ====================================================================================================

if __name__=='__main__':
    # file path
    folder_path=sys.argv[1]
    test_path_woLabel=sys.argv[2]
    output_path=sys.argv[3]

    # for now, only use business.json, user.json and trainingdata in the folder path
    train_path=folder_path+'/yelp_train.csv'
    test_path_wLabel=folder_path+'/yelp_val.csv'
    business_path=folder_path+'/business.json'
    user_path=folder_path+'/user.json'

    # ================================ Create item based prediction ================================
    # define params
    co_rated_number=30

    # create spark sc
    conf=SparkConf().\
     set("spark.executor.memory", "4g").\
     set("spark.driver.memory", "4g")
    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR") 

    # read in training and validation data
    rdd_train=sc.textFile(train_path)
    rdd_val_wLabel=sc.textFile(test_path_wLabel)
    rdd_val_woLabel=sc.textFile(test_path_woLabel)
    header=rdd_train.first()
    rdd_train=rdd_train.filter(lambda r:r!=header)
    rdd_val_wLabel=rdd_val_wLabel.filter(lambda r:r!=header)
    rdd_val_woLabel=rdd_val_woLabel.filter(lambda r:r!=header)

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
    prediction_results=rdd_val_woLabel.map(lambda r:r.split(',')[0:2]).\
                    map(lambda r:[r[0],r[1],make_predition(create_prediction_params(r[1],r[0]),r[1])]).\
                    collect()
    prediction_results=sorted(prediction_results,key=lambda r:(r[0],r[1]))
    
    # ================================ Create model based prediction ================================
    true_dict=rdd_val_wLabel.\
        map(lambda r:r.split(',')).\
        map(lambda r:[(r[0],r[1]),r[2]]).collectAsMap()

    # create additional features
    business_feature=business_features(business_path)
    user_feature=user_features(user_path)

    # create training and testing matrix
    model_training, model_testing,output_testing=create_train_test_mat(rdd_train,rdd_val_woLabel,
                                                                       business_feature,user_feature)

    # create X y
    X_train_shape,y_train_shape,X_test_shape=create_X_y(model_training,model_testing)

    # create model
    xgb_model=model(X_train_shape,y_train_shape)

    # make prediction
    predictions=xgb_model.predict(X_test_shape)
    
    # combine prediction with test lables
    model_output=[row+[p] for p, row in zip(list(predictions),output_testing)]
    model_output=sorted(model_output,key=lambda r:(r[0],r[1]))

    # ================================ create a weighted score ================================
    # get weighted pred value and true value

    truth,pred=[],[]
    final_output=[]
    for item_based,model_based in zip(prediction_results,model_output):
        weighted_score=item_based[2]*0.1+model_based[2]*0.9
        truth.append(float(true_dict[(item_based[0],item_based[1])]))
        pred.append(weighted_score)
        # create final output
        final_output.append([item_based[0],item_based[1],weighted_score])
        
    # calculate RMSE
    rmse=calculate_rmse(truth,pred)
    # output prediction files
    write_output(final_output,output_path)

    end_time=time()

    print(f'Total time: {end_time-start_time},RMSE:{rmse}')