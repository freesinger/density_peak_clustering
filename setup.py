import matplotlib.pyplot as plt

import data_process
import cluster

GIVEN_DATA = './data/example_distances.dat'

def main():
    solution = data_process.ProcessData()
    dist, maxid = solution.data_process(GIVEN_DATA)
    threshold = solution.threshold(dist, maxid)
    cutoff = solution.CutOff(dist, maxid, threshold)
    min_dist, min_num = solution.min_distance(dist, cutoff, maxid)
    pair_info, refer_info = solution.make_pair(cutoff, min_dist, maxid)
    print('Data process done!')

    clust = cluster.DensityPeakCluster()
    center, tag = clust.locate_center(refer_info, maxid)
    taginfo = clust.classify(tag, cutoff, min_num, maxid)
    print('Clustering done!')
    # print(taginfo)
    # gauss = solution.Guasse(dist, maxid, threshold)

    # show cluster distribution info
    y, x = zip(*(sorted(taginfo.items(), key=lambda k:k[1])))
    plt.scatter(x, y)
    plt.xlabel('Cluster Number')
    plt.ylabel('Point Number')
    # plt.savefig('./images/cluster.png')
    plt.show()
    
if __name__ == '__main__':
    main()
    