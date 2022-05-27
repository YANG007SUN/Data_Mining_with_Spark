from pyspark import SparkContext
import sys
import time
from collections import defaultdict
from operator import add
from itertools import permutations

start_time=time.time()

def create_adjacent_nodes(edges):
    """create dict of adjacent nodes for each node
    """
    adjacent_nodes = defaultdict(set)
    for pair in edges:
        adjacent_nodes[pair[0]].add(pair[1])
        adjacent_nodes[pair[1]].add(pair[0])
    return adjacent_nodes

def create_tree_and_nodeValue(root):
    # define dict
    tree=defaultdict(set)
    node_value=defaultdict(float)
    
    find_parent = defaultdict(set)
    bfs=defaultdict(int)
    
    # init value
    used_nodes = {root}
    children = adjacent_nodes[root]
    tree[root] = children
    bfs[root] = 1
    
    # create tree
    while children:
        new_child_nodes = set()
        used_nodes = used_nodes.union(children)
        for node in children:
            adj_nodes = adjacent_nodes[node]
            # create tree
            if adj_nodes - used_nodes:
                child_nodes = adj_nodes - used_nodes
                tree[node] = child_nodes
                
            new_child_nodes = new_child_nodes.union(adj_nodes)
            # create node value
            node_value[node]=1
        children = new_child_nodes - used_nodes
    return tree, node_value

def find_one_community(node, adjacent_nodes):
    """based on one node,find its all connected node
       node: edge with highest betweenness
    """
    used_nodes={node}
    children=adjacent_nodes[node]
    while children:
        new_child_nodes = set()
        used_nodes = used_nodes.union(children)
        for node in children:
            adj_nodes = adjacent_nodes[node]
            new_child_nodes = new_child_nodes.union(adj_nodes)
        children = new_child_nodes - used_nodes
        new_used_nodes = used_nodes.union(new_child_nodes)
    return new_used_nodes


def create_bfs(tree,root):
    """find # of shortest path for each node
    """
    # init
    bfs=defaultdict(int)
    find_parent=defaultdict(set)
    bfs[root] = 1
    
    for parent, children in tree.items():
        for child in children:
            find_parent[child].add(parent)

        for branch in tree.items():
            for val in branch[1]:
                parent_nodes = find_parent[val]
                if len(parent_nodes):
                    bfs[val] = sum([bfs[parent_node] for parent_node in parent_nodes])
                else:
                    bfs[val] = 1
    return bfs

def calculate_betweenness(tree,node_value,bfs,betweenness_dict):
    # create needed edges, loop from last branch to first
    for branch in reversed(list(tree.items())):
        temp=0
        edge=(branch[0],)
        for child in branch[1]:
            edge=(branch[0],child)

            # credit of edge depends on number of shortest path - node value/# of shortest path
            betweenness=node_value[edge[1]]*bfs[edge[0]]/bfs[edge[1]]
            temp+=betweenness
            # insert into global variable
            betweenness_dict[tuple(sorted(edge))]+=betweenness
        # update node value
        node_value[edge[0]]= node_value[edge[0]]+temp
    return betweenness_dict

def Girvan_Newman(root):
    """combine all above function
       - take one root and create tree
       - calculate betweenness for the tree
    """
    betweenness_dict=defaultdict(float)
    # create tree and node_value
    tree, node_value = create_tree_and_nodeValue(root)
    # create bfs value
    bfs=create_bfs(tree,root)
    # calculate betweeness of each edge and return the dict
    return calculate_betweenness(tree,node_value,bfs,betweenness_dict)

def create_degree_mapping(adjacent_nodes):
    """create degree mapping dict for k_i and k_j
    """
    degree_dict=defaultdict(int)
    for k,v in adjacent_nodes.items():
        degree_dict[k]=len(v)
    return degree_dict


def create_adjacent_matrix(edges):
    """dict to see if two nodes are connected
       connected:1
       not connected:0
    """
    adj_matrix=dict()
    # create all possible permu for nodes
    pairs=permutations(vertices,2)
    for pair in pairs:
        if pair in edges:
            adj_matrix[tuple(pair)]=1
        else:
            adj_matrix[tuple(pair)]=0
            
    for node in vertices:
        adj_matrix[tuple((node,node))]=0
        
    return adj_matrix
                
def find_one_community(node, adjacent_nodes):
    """based on one node,find its all connected node
       node: edge with highest betweenness
    """
    used_nodes={node}
    children=adjacent_nodes[node]
    while children:
        new_child_nodes = set()
        used_nodes = used_nodes.union(children)
        for node in children:
            adj_nodes = adjacent_nodes[node]
            new_child_nodes = new_child_nodes.union(adj_nodes)
        children = new_child_nodes - used_nodes
    return used_nodes

def find_all_communities(node, vertices,adjacent_nodes):
    """find all communities after removing each edge
       node: first node in vertices set
    """
    # store all found communities
    communities=set()
    # after finding one community based on node, all the nodes in the community have been used
    used_nodes=find_one_community(node, adjacent_nodes)
    communities.add(tuple(used_nodes))
    # pending nodes=vertices - all used nodes
    pending_nodes=vertices-used_nodes
    pointer=0
    
    while 1:
        
        community=find_one_community(list(pending_nodes)[pointer], adjacent_nodes)
        communities.add(tuple(community))
        used_nodes=used_nodes.union(community)
        
        pending_nodes=vertices-used_nodes
        
        if len(pending_nodes)==0:
            break

    return list(communities)

def calculate_modularity(communities,m):
    """calculate modularity for communities for each edge removing
    """
    mod=0
    for community in communities:
        mod+=sum(A[(i, j)] - degree[i] * degree[j] / (2 * m) for i in community for j in community)
    mod=mod/(2*m)
    return mod



if __name__ == '__main__':

    # params
    threshold=int(sys.argv[1])
    file_path=sys.argv[2]
    betweenness_outputPath=sys.argv[3]
    community_outputPath=sys.argv[4]
    
    # create sc
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('WARN')

    # read data
    rdd = sc.textFile(file_path)
    header = rdd.first()
    uid_bid = rdd.filter(lambda r: r!=header).\
                map(lambda r:(r.split(',')[0],r.split(',')[1])).\
                groupByKey().\
                map(lambda r:(r[0],list(set(r[1])))).collect()

    # create vertices and edges
    vertices=set()
    edges=set()

    for b1 in uid_bid:
        for b2 in uid_bid:
            # not the same bid
            if b1[0]!=b2[0]:
                # intersection greater and equal than threshold
                if len(set(b1[1]).intersection(set(b2[1])))>=threshold:
                    vertices.add((b1[0]))
                    vertices.add((b2[0]))
                    edges.add(tuple((b1[0],b2[0])))

    adjacent_nodes=create_adjacent_nodes(edges)

    # create betweenness
    betweenness=sc.parallelize(vertices).map(lambda node: Girvan_Newman(node)).\
                    flatMap(lambda r:list(r.items())).\
                    reduceByKey(add).\
                    map(lambda x: (x[0], round(x[1] / 2,5))).\
                    sortBy(lambda x: (-x[1], x[0])).collect()

    with open(betweenness_outputPath, 'w+') as f:
	    for pair in betweenness:
		    f.write(str(pair)[1:-1] + '\n')


    # find communities
    # create highest betweenness
    betweenness_dict=sc.parallelize(betweenness).\
                map(lambda r:(r[1],r[0])).\
                groupByKey().map(lambda r:(r[0],list(r[1]))).\
                sortBy(lambda r:-r[0]).take(1)

   # compute degree matrix
    degree=create_degree_mapping(adjacent_nodes)
    # compute adjacent matrix
    A=create_adjacent_matrix(edges)
    # total edges !!! need to divide by 2 !!!
    m=len(edges)/2
    num_edges=len(edges)/2

    max_modularity=-1


    # combine code above to compute modualarity
    while num_edges:
        pairs_to_removes=betweenness_dict[0][1]
            
        # remove edges
        for pair in pairs_to_removes:
            adjacent_nodes[pair[0]].remove(pair[1])
            adjacent_nodes[pair[1]].remove(pair[0])
            num_edges-=1
            
        # find all communities
        communities=find_all_communities(list(vertices)[0], vertices, adjacent_nodes)
        # calculate mod
        mod=calculate_modularity(communities,m)
        if mod>max_modularity:
            max_modularity=mod
            choosen_communities=communities

        # re-cal betweenness
        betweenness=sc.parallelize(vertices).map(lambda node: Girvan_Newman(node)).\
                    flatMap(lambda r:list(r.items())).\
                    reduceByKey(add).\
                    map(lambda x: (x[0], round(x[1] / 2,5))).\
                    sortBy(lambda x: (-x[1], x[0])).collect()
        
        betweenness_dict=sc.parallelize(betweenness).\
                    map(lambda r:(r[1],r[0])).\
                    groupByKey().map(lambda r:(r[0],list(r[1]))).\
                    sortBy(lambda r:-r[0]).take(1)


    sorted_communities=sorted([sorted(i) for i in choosen_communities],key=lambda r:(len(r),r))

    with open(community_outputPath, 'w+') as f:
        for community in sorted_communities:
            f.write(str(community)[1:-1] + '\n')

    end_time=time.time()
    print(f'Total time :{end_time-start_time}')