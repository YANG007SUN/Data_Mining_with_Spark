from graphframes import GraphFrame
import os
from pyspark import SparkContext
from pyspark.sql import SQLContext, functions as F
import sys
import time

start_time=time.time()
# when run locally
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"
# when run on vocareum
# os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")


if __name__ == '__main__':
    threshold = int(sys.argv[1])
    file_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # init spark
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('WARN')

    # generate structure: [user_id, [business_id1, business_id2]]
    rdd = sc.textFile(file_path)
    header = rdd.first()
    uid_bid = rdd.filter(lambda r: r!=header).\
                map(lambda r:(r.split(',')[0],r.split(',')[1])).\
                groupByKey().\
                map(lambda r:(r[0],list(set(r[1])))).collect()

    # create vertices and edges
    vertices=set()
    edges=[]

    for b1 in uid_bid:
        for b2 in uid_bid:
            # not the same bid
            if b1[0]!=b2[0]:
                # intersection greater and equal than threshold
                if len(set(b1[1]).intersection(set(b2[1])))>=threshold:
                    vertices.add((b1[0],))
                    vertices.add((b2[0],))
                    edges.append((b1[0],b2[0]))
    
    # label propagation
    vertices = SQLContext(sc).createDataFrame(list(vertices), ['id'])
    edges = SQLContext(sc).createDataFrame(edges, ['src', 'dst'])
    g = GraphFrame(vertices, edges)
    result = g.labelPropagation(maxIter=5)

    # format
    community_results=result.groupBy("label").agg(F.collect_list("id")).\
                withColumnRenamed('collect_list(id)','community')
    
    res=sorted([sorted(i['community']) for i in community_results.rdd.collect()],key=lambda r:(len(r),r))
    
    # output
    with open(output_path, 'w+') as f:
        for result in res:
            f.write(str(result).strip('[]') + '\n')
    
    end_time=time.time()
    print(f'Total time {end_time-start_time}')
