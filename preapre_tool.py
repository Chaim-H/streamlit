import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.preprocessing import StandardScaler


def data_import(size_of_data='all'):
    selection_features = [
        'linkdiameter',
        'linkslope',
        'Q_flow',
        'Q_type_uniform',
        'streamorderSH',
        'Link_residents',
        'Aspect_ratio',
        'real_density',
        'betweeness',
        'closeness',
        'current_flow_closeness',
        'second_order',
        'katz_cent',
        'harmonic_centrality',
        'degree',
    ]
    if size_of_data == 'all':
        train = pd.read_csv('data/data_of_links_train.csv')
    else:
        train = pd.read_csv('data/data_of_links_train.csv').sample(size_of_data)
    train_numpy = train[selection_features].to_numpy()
    test = pd.read_csv('data/data_of_links_test.csv')
    test_numpy = test[selection_features].to_numpy()
    return train, train_numpy, test, test_numpy, selection_features, size_of_data


def scale(train_numpy):
    scaler = StandardScaler()
    scaler.fit(train_numpy.astype(float))
    train_numpy = scaler.transform(train_numpy.astype(float))
    pickle.dump(scaler, open('StandardScaler.pkl','wb'))
    return scaler, train_numpy


def pca_for_similariy(X):
    pca = PCA(n_components=2, svd_solver='arpack')
#     pca = KernelPCA(n_components=2, kernel='linear', degree=10, gamma=0.00000001)
    # pca = KernelPCA(n_components=2, kernel='poly', gamma=0.01) ## it is worging well
    X_pca = pca.fit_transform(X)
    pickle.dump(pca, open('pca.pkl','wb'))
    return pca, X_pca

def find_train_location(X_pca, point_round=1):
    train_location = list(set([tuple(i) for i in (X_pca.round(point_round)).tolist()]))
    pickle.dump([train_location,point_round], open('train_location.pkl','wb'))
    return train_location, point_round


def figure_pca_similarity(pca, X_pca, train_location, point_round, scaler, train_numpy, test_numpy, y):
    colors = ["navy", "turquoise"]
    X_transformed = X_pca.round(point_round)
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1], ['not acc', 'acc']):
        plt.scatter(
            X_transformed[y == i, 0],
            X_transformed[y == i, 1],
            color=color,
            lw=0.1,
            alpha = 0.5,
            label=target_name,
        )
    a = test_numpy[10000:10001].copy()
    # for change in range(len(a[0])):
    #     a[0][change] = change
    for i in [9, 40000]:
        a[0][5] = i
        for j in [4,30]:
            a[0][7] = j
            for k in [1, 70]:
                a[0][4] = k
                # print(a[0][:8])
                # a[0][1] = 2
                tmp = pca.transform(scaler.transform(a))
                # tmp = pca.transform((a))
                plt.scatter(tmp[0][0].round(point_round),tmp[0][1].round(point_round), lw=10, label=f'{i}_{j}_{k}')
                if tuple(tmp[0].round(point_round)) in train_location:
                    print(f'{i}_{j}_{k:<30}: the point similar to train set')
                else:
                    print(f'{i}_{j}_{k:<30}: the point do not similar to train set')
    print(a[0])
    for i in range(10):
        a = train_numpy[i*100:i*100+1].copy()
        tmp = pca.transform(a)
        plt.scatter(tmp[0][0].round(point_round),tmp[0][1].round(point_round), lw=10)
        if tuple(tmp[0].round(point_round)) in train_location:
            print(f'{i}: the point similar to train set')
        else:
            print(f'{i}: the point do not similar to train set')

    plt.grid()
    plt.legend(loc="right", shadow=False, scatterpoints=1)
    plt.show()
    return tmp
def main():
    train, train_numpy, test, test_numpy, selection_features, _ = data_import(size_of_data=30000)
    scaler, train_numpy = scale(train_numpy)
    pca, X_pca = pca_for_similariy(X=train_numpy)
    train_location, point_round = find_train_location(X_pca, point_round=0)
    figure_pca_similarity(pca, X_pca, train_location, point_round, scaler, train_numpy, test_numpy, y=train.y_solid_acc)
    
if __name__=='__main__': 
    main()
