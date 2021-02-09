import numpy as np
import matplotlib.pyplot as plt

class ProcessData(object):
    def data_process(self, folder):
        '''
        :folder: data file path
        :rtype: dict pair distance
                MAX id number
        '''
        distance = dict()
        max_pt = 0
        with open(folder, 'r') as data:
            for line in data:
                i, j, dis = line.strip().split()
                i, j, dis = int(i), int(j), float(dis)
                distance[(i, j)] = dis
                distance[(j, i)] = dis
                max_pt = max(i, j, max_pt)
            for num in range(1, max_pt + 1):
                distance[(num, num)] = 0
        return distance, max_pt

    def entropy(self, distance, maxid, factor):
        '''
        :distance: dict with pair: dist
        :factor: impact factor
        :maxid: max elem number
        :rtype: entropy H in data field
        '''
        potential = dict()
        for i in range(1, maxid + 1):
            tmp = 0
            for j in range(1, maxid + 1):
                tmp += np.exp(-pow(distance[(i, j)] / factor, 2))
            potential[i] = tmp
        z = sum(potential.values())
        H = 0
        for i in range(1, maxid + 1):
            x = potential[i] / z
            H += x * np.log(x)
        return -H

    def threshold(self, dist, max_id):
        '''
        :rtype: factor value makes H smallest
        '''
        entro = 10.0
        # given data:
        # 0.02139999999999999 7.203581306901208
        # 0.02149999999999999 7.203577254067677
        # 0.02159999999999999 7.203577734107922

        # generate data:
        # 0.367020, 6.943842
        # 0.368959, 6.943840
        # 0.370898, 6.943841
        
        # scape = np.linspace(0.330, 0.430, 50)
        # 通用数据使用以下一行
        scape = np.linspace(0.001, 1.001, 100)
        for factor in scape:
            value = self.entropy(dist, max_id, factor)
            print('factor: {0:.6f}, entropy: {1:.8f}'.format(factor, value))
            # plt.scatter(factor, value, c='r', s=1)
            if value and value < entro:
                entro, thresh = value, factor
        thresh = 3 * thresh / pow(2, 0.5)
        
        """
        plt.xlabel(r'$\sigma$')
        plt.ylabel(r'H')
        plt.savefig('./images/Entropy test.png')
        plt.close()
        """

        print('current: ', entro, thresh)
        # given data:  7.203577254067677 0.04560838738653229
        # generate data: 6.943840312796875 0.7828967189629044
        return thresh
    
    def CutOff(self, distance, max_id, threshold):
        '''
        :rtype: list with Cut-off kernel values by desc
        '''
        cut_off = dict()
        for i in range(1, max_id + 1):
            tmp = 0
            for j in range(1, max_id + 1):
                gap = distance[(i, j)] - threshold
                tmp += 0 if gap >= 0 else 1
            cut_off[i] = tmp
        sorted_cutoff = sorted(cut_off.items(), key=lambda k:k[1], reverse=True)
        return sorted_cutoff
            
    def Guasse(self, distance, max_id, threshold):
        '''
        :rtype: list with Gaussian kernel values by desc
        '''
        guasse = dict()
        for i in range(1, max_id + 1):
            tmp = 0
            for j in range(1, max_id + 1):
                tmp += np.exp(-pow((distance[(i, j)] / threshold), 2))
            guasse[i] = tmp
        sorted_guasse = sorted(guasse.items(), key=lambda k:k[1], reverse=True)
        return sorted_guasse

    def min_distance(self, distance, srt_dens, maxid):
        '''
        :srt_dens: desc sorted list with density values (point, density)
        :rtype: min distance dict
                min number dict
        '''
        min_distance = dict()
        min_number = dict()
        h_dens = srt_dens[0][0]
        min_number[h_dens] = 0
        max_dist = -1
        for i in range(1, maxid + 1):
            max_dist = max(distance[(h_dens, i)], max_dist)
        min_distance[h_dens] = max_dist
        
        for j in range(1, len(srt_dens)):
            min_dist, min_num = 1, 0
            current_num = srt_dens[j][0]
            for k in srt_dens[0:j]:
                current_dist = distance[(current_num, k[0])]
                if current_dist < min_dist:
                    min_dist, min_num = current_dist, k[0]
            min_distance[srt_dens[j][0]] = min_dist
            min_number[current_num] = min_num
        return min_distance, min_number

    def make_pair(self, srt_dens, min_dist, maxid):
        '''
        :rtype: pair dict with {point: [density, min dist]}
                refer factor dict with {point: density * dist}
        '''
        pair_dict = dict()
        dens_dict = dict()
        refer_dict = dict()
        # convert list to dict
        for elem in srt_dens:
            dens_dict[elem[0]] = elem[1]
        if len(dens_dict) == maxid:
            for key in dens_dict.keys():
                pair_dict[key] = [dens_dict[key], min_dist[key]]
                refer_dict[key] = dens_dict[key] * min_dist[key]
        else:
            return print('missing %d value', maxid - dens_dict)
        return pair_dict, refer_dict
    
    def show_pair_info(self, pair, threshold):
        show_dict = dict()
        for p in pair.values():
            show_dict[p[0]] = p[1]
        tmp = sorted(show_dict.items())
        dens, mdis = zip(*tmp)
        plt.scatter(dens, mdis)
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$\delta$')
        plt.title(r'$d_c=$' + str(threshold))
        plt.savefig('./images/Decision Graph Cutoff test.png')
        plt.close()
