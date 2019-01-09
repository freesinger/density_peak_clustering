import matplotlib.pyplot as plt

import data_process
import cluster

TEST_DATA = './data/generatePoints_distance.txt'
COOR_DATA = './data/generatePoints.txt'

def main():
    solution = data_process.ProcessData()
    dist, maxid = solution.data_process(TEST_DATA)
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
    print(center)   # [80, 460, 463]

    # show each cluster results
    clust.analysis(center, taginfo, dist, maxid)
    
    # show cluster distribution info
    temp = sorted(taginfo.items(), key=lambda k:k[1])
    
    with open(COOR_DATA, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        coords = dict()
        for line in lines:
            p, x, y = line.strip().split()
            p, x, y = int(p), float(x), float(y)
            coords[p] = [x, y]
    # print(coords[center[0]])
    for i in range(len(center) - 1):
        c = coords[center[i]]
        plt.plot(c[0], c[1], 'ok', markersize=5, alpha=0.8)
    
    color = {0:'k', 1:'b', 2:'g', 3:'r', 4:'c', 5:'m', 6:'y'}
    for p in temp:
        for i in range(len(center)):
            try:
                c = coords[p[0]]
                if p[1] == i:
                    plt.scatter(c[0], c[1], c=color[i], alpha=0.6, s=1)
            except KeyError:
                continue
                # plt.scatter(c[0], c[1], c=color[i], alpha=0.6, s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot Result')
    plt.savefig('./images/result.png')
    plt.show()

    """
    y, x = zip(*temp)
    plt.scatter(x, y)
    plt.xlabel('Cluster Number')
    plt.ylabel('Point Number')
    plt.title(r'$d_c=$' + str(threshold))
    plt.savefig('./images/cluster_cutoff_test.png')
    plt.show()
    """
    
if __name__ == '__main__':
    main()
    