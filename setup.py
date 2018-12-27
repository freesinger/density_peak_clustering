import matplotlib.pyplot as plt

import data_process
import cluster

GIVEN_DATA = './data/example_distances.dat'

def main():
    solution = data_process.ProcessData()
    dist, maxid = solution.data_process(GIVEN_DATA)
    threshold = solution.threshold(dist, maxid)
    sort_dst = solution.CutOff(dist, maxid, threshold)
    # sort_dst = solution.Guasse(dist, maxid, threshold)
    min_dist, min_num = solution.min_distance(dist, sort_dst, maxid)
    pair_info, refer_info = solution.make_pair(sort_dst, min_dist, maxid)
    solution.show_pair_info(pair_info, threshold)
    print('Data process done!')

    clust = cluster.DensityPeakCluster()
    center, tag = clust.locate_center(refer_info, maxid, threshold)
    taginfo = clust.classify(tag, sort_dst, min_num, maxid)
    print('Clustering done!')
    # print(taginfo)
    # gauss = solution.Guasse(dist, maxid, threshold)

    # show cluster distribution info
    temp = sorted(taginfo.items(), key=lambda k:k[1])
    y, x = zip(*temp)
    # color = { 1:'b', 2:'g', 3:'r', 4:'c', 5:'m', 6:'y'}
    plt.scatter(x, y)
    plt.xlabel('Cluster Number')
    plt.ylabel('Point Number')
    plt.title(r'$d_c=$' + str(threshold))
    plt.savefig('./images/cluster_cutoff.png')
    plt.show()
    
if __name__ == '__main__':
    main()
    