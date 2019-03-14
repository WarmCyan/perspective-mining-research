import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

as_vec = np.asarray([
    [1,0,0,-1,.5],
    [-1,0,0,1,-.75],
    [.5,0,1,0,.1],
    [.7,-.3,0,0,0],
    [.1,-.1,.1,.1,.1]])

sim_mat = cosine_similarity(as_vec)

ap_vec = np.asarray([
    [1,0,0,1,1],
    [1,0,0,1,1],
    [1,0,1,0,1],
    [1,1,0,0,0],
    [1,1,1,1,1]])

sim_mat_ap = cosine_similarity(ap_vec)
