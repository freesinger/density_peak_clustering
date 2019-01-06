import math
import numpy as np
import matplotlib.pyplot as plt

class DensityPeakCluster(object):
    def locate_center(self, judge, maxid, threshold):
        '''
        :judge: dict with {point: density * dist}
        :rtype: list of cluster centers
                dict of tag info
        '''
        
        result = sorted(judge.items(), key=lambda k:k[1], reverse=True)
        
        x, y = zip(*result)
        plt.scatter(x, y)
        plt.xlabel('Point Number')
        plt.ylabel(r'$\gamma$')
        plt.title(r'$d_c=$'+ str(threshold))
        plt.savefig('./images/rank cutoff.png')
        # plt.show()
        plt.close()
        # result showed in rank.png
        # 6 clusters should be divided

        cluster_centers = list(c[0] for c in result[0:6])
        # [1061, 1515, 400, 6, 1566, 614]

        tag_info = dict()
        cluster_id = 1
        for i in range(1, maxid + 1):
            if i in cluster_centers:
                tag_info[i] = cluster_id
                cluster_id += 1
            else:
                tag_info[i] = -1

        return cluster_centers, tag_info

    def classify(self, taginfo, srt_dens, min_num, maxid):
        '''
        :rtype: tag dict with classified points not cluster center
        '''
        dens_dict = dict()
        for ele in srt_dens:
            dens_dict[ele[0]] = ele[1]
        for i in dens_dict.keys():
            if taginfo[i] == -1:
                taginfo[i] = taginfo[min_num[i]]
        return taginfo

    def analysis(self, centers, taginfo, distance, maxid):
        '''
        :rtype: plot cluster information
        '''
        num_centers = len(centers)
        tmp = sorted(taginfo.items(), key=lambda k:k[1])
        dvid_numbers = list()
        for i in range(1, num_centers + 1):
            cluster_i = list()
            for pair in tmp:
                if pair[1] == i:
                    cluster_i.append(pair[0])
            dvid_numbers.append(cluster_i)
        
        for i in range(1, num_centers + 1):
            cur_set = dvid_numbers[i - 1]
            d = list(distance[(j, i)] for j in cur_set)
            plt.stackplot(cur_set, d)
            plt.xlabel('Point Number')
            plt.ylabel('Distance to Center')
            plt.title('Cluster No.{}'.format(i))
            plt.savefig('./images/Cluster{}'.format(i))
            plt.close()
            