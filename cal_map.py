# -- coding: utf-8 --
from torch.autograd import Variable
import numpy as np

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1]  # max inner product value
    # distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    distH = 0.5 * (q - np.dot(B1, B2.T))
    return distH

# Calculate the map
def calculate_map(qB, rB, queryL, retrievalL):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.shape[0]
    map = 0
    for iter in range(num_query):
        # gnd : check if exists any retrieval items with same label
        gnd = (retrievalL.cpu() == queryL[iter].cpu()).numpy().astype(np.float32)

        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        # sort gnd by hamming dist
        nn = qB[iter, :].cpu().numpy()
        mm = rB.cpu().numpy()
        hamm = calculate_hamming(nn, mm)

        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))

        map = map + map_
    map = map / num_query
    return map

# Calculate the map of the returned topk data
def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        if iter % 100 == 0:
            print("query: ", iter)
        # gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        gnd = (retrievalL.cpu() == queryL[iter].cpu()).numpy().astype(np.float32)
        nn = qB[iter, :].cpu().numpy()
        mm = rB.cpu().numpy()
        hamm = calculate_hamming(nn, mm)

        ind = np.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)

        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_

    topkmap = topkmap / num_query
    return topkmap

# Calculate the map with the Hamming radius of 2
def calculate_map_radius(qB, rB, queryL, retrievalL, r=2):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.shape[0]
    toprmap = 0
    for iter in range(num_query):
        # gnd : check if exists any retrieval items with same label
        gnd = (retrievalL.cpu() == queryL[iter].cpu()).numpy().astype(np.float32)
        # sort gnd by hamming dist
        nn = qB[iter, :].cpu().numpy()
        mm = rB.cpu().numpy()
        hamm = calculate_hamming(nn, mm)

        ind = np.argsort(hamm)
        topk = np.sum(hamm <= r)

        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)

        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        toprmap = toprmap + topkmap_

    toprmap = toprmap / num_query
    return toprmap

# Calculate the precision with the Hamming radius of 2
def calculate_pre_radius(qB, rB, queryL, retrievalL, r=2):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.shape[0]
    toprmap = 0
    for iter in range(num_query):
        # gnd : check if exists any retrieval items with same label
        gnd = (retrievalL.cpu() == queryL[iter].cpu()).numpy().astype(np.float32)
        # sort gnd by hamming dist
        nn = qB[iter, :].cpu().numpy()
        mm = rB.cpu().numpy()
        hamm = calculate_hamming(nn, mm)

        ind = np.argsort(hamm)
        topk = np.sum(hamm <= r)

        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        topkmap_ = tsum / (topk+1e-15)

        toprmap = toprmap + topkmap_

    toprmap = toprmap / num_query
    return toprmap

# Calculate the precision of the returned topk data
def calculate_pre_topk(qB, rB, queryL, retrievalL, topk):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        # gnd : check if exists any retrieval items with same label
        gnd = (retrievalL.cpu() == queryL[iter].cpu()).numpy().astype(np.float32)
        # sort gnd by hamming dist
        nn = qB[iter, :].cpu().numpy()
        mm = rB.cpu().numpy()
        hamm = calculate_hamming(nn, mm)

        ind = np.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)

        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_

    topkmap = topkmap / num_query
    return topkmap


