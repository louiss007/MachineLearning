from sim import cosSim
from numpy import *

def loadExData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:, item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], \
                                 dataMat[overLap,j])
        #print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, eatMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = eatMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, \
                  key=lambda jj:jj[1], reverse=True)[:N]

#in the mode of command line
##myMat=np.mat(svdRec.loadExData())
myMat=mat(loadExData())
lis=recommend(myMat, 2)

