# ======================================================
#  This script is for creating model and outout 
#  xgb model called 'model.md'
# ======================================================

from pyspark import SparkContext, SparkConf
from collections import defaultdict
import sys
import csv
import math
import json
import xgboost as xgb
from joblib import dump, load
from time import time
import numpy as np

start_time=time()

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

    # create sc session
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

    # true label dictionary for calculate RMSE with prediction
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

    # output model
    dump(xgb_model,'model.md')


    

