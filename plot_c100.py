import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os


import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import leastsq


def Fun2(p,x):                        
    a1,a2,a3,a4 = p
    return a1*x*x*x + a2*x*x + a3*x + a4

def error (p,x,y):                  
    return Fun2(p,x)-y 


def plot(results, labels, title=None, save_path='1.pdf', ylim=None, xlim=None, fontsize=None,legend_flag = 0, ylabel = 'Accuracy', xlabel = None, x_axis = None):

    for i, (result, label) in enumerate(zip(results, labels)):
        mean = np.mean(result, axis=0)
        # print(result.shape)
        std = np.std(result, axis=0)
        # print(mean)
        _mean = gaussian_filter1d(mean, sigma=3)
        _min = gaussian_filter1d(mean - std, sigma=3)
        _max = gaussian_filter1d(mean + std, sigma=3)
        if x_axis is None:
            if label == 'Average':
                plt.plot(_mean, label=label, linewidth = 2, color=f'C{i}')
            else:
                plt.plot(_mean, label=label, linestyle = '--', color=f'C{i}')
            # print(label)
            plt.fill_between(range(mean.size), _min, _max, color=f'C{i}', alpha=0.3)
        else:
            x_mean = np.mean(x_axis, axis=0)
            plt.plot(x_mean, _mean, label=label, linestyle = '--', color=f'C{i}')
            # print(label)
            plt.fill_between(x_mean, _min, _max, color=f'C{i}', alpha=0.3)

    if title:
        plt.title(title,fontsize=fontsize)
    if xlabel is None:
        plt.xlabel(f"Epochs (x{INTERVAL})", fontsize=fontsize)
    else:
        plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
        # if legend_flag:
    # plt.legend(loc=3,fontsize=fontsize)
    plt.grid(color='gray', linestyle='-.', linewidth=0.5)
    if ylim:
        plt.ylim(ylim)
    # plt.xlim(xlim)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # plt.tight_layout()
        # pdf.savefig()
    # plt.savefig(save_path)
    # plt.show()





def chop(arr):
    """ chop an 1d array into a 2d array """
    return np.array([arr[i:i+INTERVAL] for i in range(len(arr)-INTERVAL)]).T

def preprocess_csv(root,file_name):
    result=np.array(pd.read_csv(f'{root}/{file_name}.csv', header = None)) # remember to change the file path
    with open(f'{root}/{file_name}.pickle', 'wb') as handle:
        pickle.dump(result, handle)

def get_acc(root, file_name, dataset):
    
    with open(f'{root}/{file_name}.pickle', 'rb') as handle:
        result = pickle.load(handle)
    if dataset == 'c10':
        per_class_acc = np.array(result[:-1,6:16], dtype = np.float_)
    else:
        per_class_acc = np.array(result[:-1,6:106], dtype = np.float_)
    # epoch = result[:-1,1]
    average = np.array(result[:-1,-1], dtype = np.float_)
    # print(f'Best average test acc is {np.max(average)} \nPer class acc is \n{result[np.argmax(average),6:106]}')
    acc = per_class_acc[np.argmax(average)]
    return acc

    
# --------------    preprocess data    ----------------


acc_small, acc_ssl, acc_full = None, None, None
num = 1000 # 500 1000 2000 4000
fig=plt.figure(figsize=(15,4))
cnt = 1 
plt.subplots_adjust(hspace=0, wspace=-1)
for num in [500, 1000]:
    for alg in ['mm', 'uda']:
        root = f'{alg}/result_c100'
        files = os.listdir(root)

        ax = fig.add_subplot(1, 4, cnt)
        cnt += 1



        for method in ['small', 'ssl', 'full']:
            acc_rec = []
            for file_i in files:
                if method == 'full':
                    # name = f'mm-{method}'
                    name = f'{alg}-{method}'
                else:

                    name = f'{alg}-{method}-{num}'
                if name in file_i:
                    fp =open(os.path.join(root,file_i))
                    acc = []
                    avg = []
                    for line in fp.readlines(): 
                        data = ' '.join(line.replace('[','').replace(']','').replace(',',' ').split()).split(' ')
                        start = data.index('is:') + 1
                        tmp = np.array(data[start:start+100])
                        avg += [float(data[-1])]
                        acc.append([float(i) for i in tmp])
                    if len(acc) > 0:
                        loc = np.argmax(avg)
                        # acc = acc[-5:]
                        # acc = np.mean(acc[-5:],axis=0)
                        if method == 'small' or alg == 'mm':
                            acc = np.mean(acc[-10:],axis=0)
                            # acc_rec.append(acc)
                        else:
                            acc = np.mean(acc[loc-2:loc+2],axis=0)
                            # acc = acc[loc-1:loc+1]
                            # acc_rec += acc
                        acc_rec.append(acc)
                        # print(acc)
                        
                    fp.close()
            # exec(f'acc_{method} = np.mean(acc_rec,axis=0)')
            exec(f'acc_{method} = np.array(acc_rec)')

        acc_small_rec = []
        benefit_rec = []
        acc_ssl_rec = []
        thre = 0
        for i in range(acc_small.shape[0]):
            for j in range(acc_ssl.shape[0]):
                for k in range(acc_full.shape[0]):
                    acc_small_tmp = acc_small[i] 
                    benefit_rec += [(acc_ssl[j] - acc_small_tmp + thre)/(acc_full[k] - acc_small_tmp + thre)]

                    acc_small_rec += [acc_small_tmp]
                    acc_ssl_rec += [acc_ssl[j]]

                
        a_s = np.array(benefit_rec).reshape(-1)
        acc_small = np.array(acc_small_rec).reshape(-1)
        acc_ssl = np.array(acc_ssl_rec).reshape(-1)



        a_s[a_s<-1] = 0.5
        a_s[a_s>2] = 0.5
        
        dataset = 'c100'

        INTERVAL = 10 if dataset == 'c10' else 400




        mean_small = np.mean(acc_small)
        mean_full = np.mean(acc_full)
        mean_ssl = np.mean(acc_ssl)

        std_small = np.std(acc_small)
        std_full = np.std(acc_full)
        std_ssl = np.std(acc_ssl)

        a_l = acc_small # train with a small labeled set




        # least-square fitting, second-order
        x = a_l
        y = a_s

        x = np.array(x)
        y = np.array(y)



        p0 = [-0.1,0.1,0.1,0.1] 
        para =leastsq(error, p0, args=(x,y))
        num_bins = 10
        _, val = np.histogram(x, bins = num_bins)
        x_new = []
        y_new = []
        for i in range(num_bins-1):  
            loc = (x>val[i]) * (x<val[i+1])
            x_tmp = x[loc]
            data = y[loc]
            mean = Fun2(para[0],np.mean(x_tmp))
            # mean = np.mean(data)
            std = np.std(data)
            loc1 = data>mean+1*std  # remove outliers
            loc2 = data<mean-1*std
            data[loc1] = mean
            data[loc2] = mean
            x_tmp[loc1] = np.mean(x_tmp)
            x_tmp[loc2] = np.mean(x_tmp)
            x_new += x_tmp.tolist()
            y_new += data.tolist()


        x = np.array(x_new)
        y = np.array(y_new)


        p0 = [-0.1,0.1,0.1,0.1] 
        para =leastsq(error, p0, args=(x,y))
        y_fitted = Fun2(para[0],x)


        benefit_ratio_mean = np.mean(y)
        benefit_ratio_std = np.std(y)


        yy = [chop(y[ (np.argsort(x)) ])]

        name = 'MixMatch' if alg == 'mm' else 'UDA'
        title = name + f', labeled size {num//100}$\\times$5$\\times$20'

        plot( yy, ['1'], x_axis = chop(np.sort(x)), xlabel = f'Accuracy with {num} clean labels', ylabel = 'Benefit Ratio', title = title, save_path=f'fig3_{alg}_{num}.pdf', fontsize = 14)
# print(np.round(mean_ssl,2))
plt.tight_layout()
plt.savefig('figure_c100.pdf')
# plt.show()