from pyspark import SparkConf,SparkContext
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from math import sqrt
import sys
import time


start_time=time.time()
# function
def create_kmeansModel(data_points,n_clusters):
    """return a kmeans model
       data points: [[index, features]]
    """
    X_features=[i[1] for i in data_points]
    X = np.array(X_features)
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    prediction=kmeans.predict(X)
    
    return kmeans, prediction

def create_RSorDS_Dict(kmeans_label,data_points)->dict:
    """create dict to see which points have cluster of 1
    """
    RSDS_dict=defaultdict(list)
    
    RSDS_candidate=[(k,v) for k,v in zip(kmeans_label,data_points)]
    RSDS_dict_list=sc.parallelize(RSDS_candidate).groupByKey().map(lambda r:(r[0],list(r[1]))).collect()
            
    # create dict
    for i in RSDS_dict_list:
        RSDS_dict[i[0]]=i[1]
                
    return RSDS_dict

def create_RS(RS_dict):
    """return RS list. (RS points are those with only 1 cluster)
    """
    RS=[v[0] for k,v in RS_dict.items() if len(v)==1]

    return RS

def create_DSorCSset_centriod_variance(DS_dict):
    """create DS set:[N,SUM,SUMSQ]
       centriods, and variances for all dimensions
    """
    DS=[]
    centriods=[]
    variances=[]
    grouping_id=[]

    for k in DS_dict.keys():
        N=len(DS_dict[k])
        # cal SUM and SUMSQ
        X_features=[]
        temp=[]
        for i in range(N):
            X_feature=DS_dict[k][i][1]
            X_features.append(X_feature)
            point_idx=DS_dict[k][i][0]
            temp.append(point_idx)

        SUM=list(np.array(X_features).sum(axis=0))
        pt_sqr=np.array(X_features)**2
        SUMSQ=list(pt_sqr.sum(axis=0))
        # DS set
        DS.append([N,SUM,SUMSQ])
        # grouping
        grouping_id.append(temp)
        # centriods
        centriods.append([i/N for i in SUM])
        # variances
        numerator=[i/N for i in SUMSQ]
        denominator=[(i/N)**2 for i in SUM]
        variances.append([n-d for n,d in zip(numerator,denominator)])
    return DS,grouping_id,centriods,variances

def update_CS_info(RS_dict,CS,cs_centriods,cs_grouping_id):
    """func does not return any,but update cs information
    """
    for k in RS_dict.keys():
        if len(RS_dict[k])>1:
            N=len(RS_dict[k])
            X_features=[]
            temp=[]
            for i in range(N):
                X_feature=RS_dict[k][i][1]
                X_features.append(X_feature)
                point_idx=RS_dict[k][i][0]
                temp.append(point_idx)

            SUM=list(np.array(X_features).sum(axis=0))
            pt_sqr=np.array(X_features)**2
            SUMSQ=list(pt_sqr.sum(axis=0))
            # DS set
            CS.append([N,SUM,SUMSQ])
            # grouping
            cs_grouping_id.append(temp)
            # centriods
            cs_centriods.append([i/N for i in SUM])
            
def create_intermminate_results(intermminate_results, round_num):
    """keep appending data to intermminate_results
    """
    # round 
    r=f'Round {round_num}'
    
    # number of DS points
    ds_point_count=0
    for i in DS:
        ds_point_count+=i[0]
    
    # number of clusters in the CS
    cs_cluster=len(CS)
    
    # number of CS points
    cs_point_count=0
    for i in CS:
        cs_point_count+=i[0]
    
    # number of RS points
    rs_point_count=len(RS)
    
    intermminate_results.append([r,ds_point_count,cs_cluster,cs_point_count,rs_point_count])
    

def calculate_std(SUM,SUMSQ,N):
    """cal std to M Distance 
       return list of std for d dimensions
    """
    numerator=[i/N for i in SUMSQ]
    denominator=[(i/N)**2 for i in SUM]
    std=[sqrt(n-d) for n,d in zip(numerator,denominator)]
    
    return std

def calculate_Mdistance(point,ds_centriods,DS):
    """calculate MD for one point to all clusters in DS
       return list of MD for point to c clusters
    """
    MD_list=[]
    for centriod, stats in zip(ds_centriods,DS):
        # calculate std
        stds=calculate_std(stats[1],stats[2],stats[0])
        MD=sqrt(sum([((x-c)/std)**2 for x,c,std in zip(point,centriod,stds)]))
        MD_list.append(MD)
    return MD_list

def update_DS_info(twenty_percent_data):
    """for processing each twenty percent data, either add point to DS or to RS_CS_list for further processing
       updating DS, ds_centriods, ds_grouping_id
    """
    RS_CS_list=[]
    for point in twenty_percent_data:
        X_features=point[1]
        pt_idx=point[0]
        # calculate MD for each point to DS stats
        MD_list=calculate_Mdistance(X_features,ds_centriods,DS)
        # find min MD distance index number
        min_idx=np.argmin(MD_list)
        # if min MD distance < 2* sqrt(dimension) then assign the point to DS
        if min(MD_list)<2*sqrt(len(X_features)):
            # update DS set [N,SUM,SUMSQ]
            DS[min_idx][0]+=1
            DS[min_idx][1]=[i+j for i,j in zip(DS[min_idx][1],X_features)]
            DS[min_idx][2]=[i+j**2 for i,j in zip(DS[min_idx][2],X_features)]
            # update ds_centroids
            ds_centriods[min_idx]=[i/DS[min_idx][0] for i in DS[min_idx][1]]
            # add point index to grouping_id
            ds_grouping_id[min_idx].append(pt_idx)
        else:
            RS_CS_list.append(point)
    return RS_CS_list

def merge_CStoDS(CS,cs_centriods,cs_grouping_id):
    """in the last run, add all qualified CS to DS
    """
    CS=[]
    cs_centriods=[]
    cs_grouping_id=[]
    d=len(DS[0][1])
    for cs_stat,cs_centriod,cs_pts in zip(CS,cs_centriods,cs_grouping_id):
        # cal MD distance
        MD_list=calculate_Mdistance(cs_centriod,ds_centriods,ds_grouping_id)
        pos_of_update=np.argmin(MD_list)
        if min(MD_list)<2*sqrt(len(d)):
            # update DS
            for pt in cs_pts:
                # DS-N
                DS[pos_of_update][0]+=1
                # DS-SUM
                DS[pos_of_update][1]=[i+j for i,j in zip(DS[pos_of_update][1], twenty_percent_data[pt][1])]
                # DS-SUMSQ
                DS[pos_of_update][2]=[i+j**2 for i,j in zip(DS[pos_of_update][1], twenty_percent_data[pt][2])]
                # ds_grouping_id
                ds_grouping_id[pos_of_update].append(pt)
        else:
            CS.append(cs_stat)
            cs_centriods.append(cs_centriod)
            cs_grouping_id.append(cs_pts)
            
    return CS,cs_centriods,cs_grouping_id


if __name__=='__main__':

    # input paras
    file_path=sys.argv[1]
    n_cluster=int(sys.argv[2])
    output_path=sys.argv[3]

    # set sc
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('WARN')

    # read data
    # only take index and feature points (not true label)
    data=sc.textFile(file_path).map(lambda r:r.split(',')).map(lambda r:(r[0],[float(i) for i in r[2:]])).collect()

    twenty_percent=int(len(data)*.2)
    collections=[data[i*twenty_percent:twenty_percent*(i+1)] for i in range(5)]

    # step1
    # Load 20% of the data randomly.
    first_twenty_percent=collections[0]

    # step2
    # Run K-Means (e.g., from sklearn) with a large K 
    # (e.g., 5 times of the number of the input clusters) on the data in memory 
    # using the Euclidean distance as the similarity measurement.
    kmeans_first, prediction=create_kmeansModel(first_twenty_percent, n_cluster*5)

    # step3
    # In the K-Means result from Step 2, move all the clusters that contain only one point to RS (outliers).
    RS_dict=create_RSorDS_Dict(prediction,first_twenty_percent)
    RS=create_RS(RS_dict)
    # remove RS data points
    for dp in RS:
        first_twenty_percent.remove(dp)
    
    # step4
    # Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
    kmeans,predictions=create_kmeansModel(first_twenty_percent,n_cluster)

    # step5
    # Use the K-Means result from Step 4 to generate the DS clusters 
    # (i.e., discard their points and generate statistics).
    DS_dict=create_RSorDS_Dict(predictions,first_twenty_percent)
    DS,ds_grouping_id,ds_centriods,ds_variances = create_DSorCSset_centriod_variance(DS_dict)

    # step6
    # Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input clusters) 
    # to generate CS (clusters with more than one points) and RS (clusters with only one point).
    if len(RS)<=n_cluster:
        new_cluster=int(round(n_cluster/2,0))
    else:
        new_cluster=n_cluster
    kmeans,predictions=create_kmeansModel(RS,new_cluster)
    RS_dict=create_RSorDS_Dict(predictions,RS)
    RS=create_RS(RS_dict)

    # init cs stats
    CS=[]
    cs_centriods=[]
    cs_grouping_id=[]

    update_CS_info(RS_dict,CS,cs_centriods,cs_grouping_id)

    # first ouput file
    intermminate_results = []
    create_intermminate_results(intermminate_results,1)

    # step 7 - 12
    i=1
    while i<5:
        # step 7
        # Load another 20% of the data randomly.
        twenty_percent_data=collections[i]
        # step 8 - 10
        RS_CS_list=update_DS_info(twenty_percent_data)
        # assigin all point from RS_CS_list to RS
        RS=RS_CS_list+RS
        # step 11
        # Run K-Means on the RS with a large K 
        # (e.g., 5 times of the number of the input clusters) to generate CS 
        # (clusters with more than one points) and RS (clusters with only one point).
        if len(RS)<=n_cluster:
            new_cluster=int(round(n_cluster/2,0))
        else:
            new_cluster=n_cluster
        kmeans,predictions=create_kmeansModel(RS,new_cluster)
        RS_dict=create_RSorDS_Dict(predictions,twenty_percent_data)
        RS=create_RS(RS_dict)
        update_CS_info(RS_dict,CS,cs_centriods,cs_grouping_id)
        
        
        i+=1
        if i==5:
            CS,cs_centriods,cs_grouping_id=merge_CStoDS(CS,cs_centriods,cs_grouping_id)
        create_intermminate_results(intermminate_results,i)

    clustering_res = []
    for i, cluster in enumerate(ds_grouping_id):
        for idx in cluster:
            clustering_res.append((idx,i))
    for cluster in cs_grouping_id:
        for idx in cluster:
            clustering_res.append((idx,-1))
    for id_pt in RS:
        clustering_res.append((id_pt[0],-1))
    clustering_res = sorted(clustering_res, key = lambda pair : int(pair[0]))

    # output file
    with open(output_path,"w+") as f:
        f.write('The intermediate results:\n')
        for res in intermminate_results:
            f.write(str(res[0])+': '+str(res[1])+','+str(res[2])+','+str(res[3])+','+str(res[4])+'\n')
        f.write('\n')
        f.write('The clustering results:\n')
        for pair in clustering_res:
            f.write(str(pair[0])+','+str(pair[1])+'\n')


end_time=time.time()
print(f'Total time: {end_time-start_time}')