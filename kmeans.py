from typing import List
from scratch.linear_algebra import vector_mean
from scratch.linear_algebra import vector
def num_deference(v1:vector,v2:vector)->int:
    assert len(v1)==len(v2)
    return len([x1 for x1,x2 in zip(v1,v2) if x1!=x2])
def cluster_means(k:int,inputs:list[vector],assignments:List[int])->List[vector]:
    clusters=[[]for i in range (k)]
    for input.assignment in zip(inputs,assignments):
        clusters[assignment].append(input)
    return [vector_mean(cluster)if cluster else random.choice(inputs) for cluster in clusters]
