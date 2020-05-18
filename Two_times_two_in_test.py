# _*_ coding:utf-8 _*_
# written by Naoki Setoguchi
# date 28/08/2018

import chainer
from chainer import functions as F
from chainer import training, reporter, iterators, optimizers, serializers, Variable
from chainer.training import extensions
from chainer.backends import cuda
import chainer.computational_graph as c
import numpy as np
import glob
import math
import random
import os
# from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append(os.getcwd() + '/4inputs')
from Two_times_two_in import two_times_two_in
from loader import n_fold_cross_validation, load_usf
import argparse

def primitive_extract_features(model, gl1, gl2, pr1, pr2, id_list):

    # galleyのquery対probeの全データの相違度を計算
    correct = 0.0
    uncorrects = []
    for g1, g2, p1, p2, id in zip(gl1, gl2, pr1, pr2, id_list):

        query1 = g1.reshape([1, 3, 128, 88])
        query2 = g2.reshape([1, 3, 128, 88])
        query_id = id

        # tuple(data. id)を分割
        # p_data, p_id = map(list, zip(*probe))
        p1_data = np.asarray(pr1, dtype=np.float32)
        p1_data = p1_data.reshape([p1_data.shape[0], 3, 128, 88])
        p2_data = np.asarray(pr2, dtype=np.float32)
        p2_data = p2_data.reshape([p2_data.shape[0], 3, 128, 88])
        p_id = np.asarray(id_list, dtype=np.int32)

        # query対全データの相違度を計算する
        # queryデータをデータ数分複製して，まとめて計算
        queries1 = query1
        queries2 = query2

        num = 1.0
        while num < len(gl1):
            queries1 = np.vstack((queries1, query1))
            queries2 = np.vstack((queries2, query2))
            num += 1

        q1_sub1, q1_sub2 = np.array_split(queries1, [int(math.ceil(queries1.shape[0] / 2.0))], axis=0)
        q2_sub1, q2_sub2 = np.array_split(queries2, [int(math.ceil(queries2.shape[0] / 2.0))], axis=0)
        p1_sub1, p1_sub2 = np.array_split(p1_data, [int(math.ceil(p1_data.shape[0] / 2.0))], axis=0)
        p2_sub1, p2_sub2 = np.array_split(p2_data, [int(math.ceil(p2_data.shape[0] / 2.0))], axis=0)

        p_out, n_out = forward(q1_sub1, q2_sub1, p1_sub1, p2_sub1, model)
        tmp1, tmp2 = forward(q1_sub2, q2_sub2, p1_sub2, p2_sub2, model)

        p_out = np.vstack((p_out, tmp1))
        n_out = np.vstack((n_out, tmp2))

        # L2ノルムを計算
        diff = p_out - n_out
        dist_sq = F.sum(diff**2, axis=1)
        dist = F.sqrt(dist_sq)**2
        # l2_norm = F.batch_l2_norm_squared(p_out-n_out)
        # dissimilarity = l2_norm.data ** 2
        dissimilarity = dist.data
        dissimilarity = dissimilarity.reshape([dissimilarity.shape[0], 1])

        # 相違度が一番小さいインデックスを返す
        mini_idx = cuda.to_cpu(dissimilarity.argmin(axis=0))
        max_idx = cuda.to_cpu(dissimilarity.argmax(axis=0))

        # 相違度が最も小さいペアのidが同一人物なら，正解とカウント

        if query_id == p_id[mini_idx]:
            correct += 1.0
            print(query_id, p_id[mini_idx], 'dissilarity:{}'.format(dissimilarity[mini_idx]), 'max:{}'.format(dissimilarity[max_idx]))
        else:
            # print query_id, p_id[mini_idx], 'dissilarity:{}'.format(dissimilarity[mini_idx]), 'max:{}'.format(dissimilarity[max_idx])
            uncorrects.append(query_id)

    rate = correct / len(gl1)*100.0
    return rate, uncorrects


def forward(queries1, queries2, p1_data, p2_data, model, gpu_id):

    queries1 = cuda.to_gpu(queries1, gpu_id)
    queries2 = cuda.to_gpu(queries2, gpu_id)
    p1_data = cuda.to_gpu(p1_data, gpu_id)
    p2_data = cuda.to_gpu(p2_data, gpu_id)

    # Forward
    with chainer.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            p_out, n_out = model(queries1, queries2, p1_data, p2_data)
    p_out = cuda.to_cpu(p_out.data)
    n_out = cuda.to_cpu(n_out.data)

    return p_out, n_out


def extract_features(model, gl1, gl2, pr1, pr2, id_list, gpu_id):
    g1_data = np.asarray(gl1, dtype=np.float32)
    g1_data = g1_data.reshape([g1_data.shape[0], 3, 128, 88])
    g2_data = np.asarray(gl2, dtype=np.float32)
    g2_data = g2_data.reshape([g2_data.shape[0], 3, 128, 88])
    g_id = np.asarray(id_list, dtype=np.int32)

    p1_data = np.asarray(pr1, dtype=np.float32)
    p1_data = p1_data.reshape([p1_data.shape[0], 3, 128, 88])
    p2_data = np.asarray(pr2, dtype=np.float32)
    p2_data = p2_data.reshape([p2_data.shape[0], 3, 128, 88])
    p_id = np.asarray(id_list, dtype=np.int32)

    q1_sub1, q1_sub2 = np.array_split(g1_data, 2, axis=0)
    q2_sub1, q2_sub2 = np.array_split(g2_data, 2, axis=0)
    p1_sub1, p1_sub2 = np.array_split(p1_data, 2, axis=0)
    p2_sub1, p2_sub2 = np.array_split(p2_data, 2, axis=0)
    # q1_sub1, q1_sub2 = np.array_split(g1_data, 2.0, axis=0)
    # q2_sub1, q2_sub2 = np.array_split(g2_data, 2.0, axis=0)
    # p1_sub1, p1_sub2 = np.array_split(p1_data, 2.0, axis=0)
    # p2_sub1, p2_sub2 = np.array_split(p2_data, 2.0, axis=0)

    p_out, n_out = forward(q1_sub1, q2_sub1, p1_sub1, p2_sub1, model, gpu_id)
    tmp1, tmp2 = forward(q1_sub2, q2_sub2, p1_sub2, p2_sub2, model, gpu_id)

    gallery = np.vstack((p_out, tmp1))

    probe = np.vstack((n_out, tmp2))

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(gallery, g_id)

    correct = 0.0
    for i, item in enumerate(probe):
        predit = neigh.predict([item])[0]
        # print "label:%d, predict:%d" % (test_labels1[i], predit)
        if predit == g_id[i]:
            correct = correct + 1
    # else:
    #         f.write("label:%d, predict:%d" %(true_label[i],predit)+"\n")

    acc = correct / len(probe)

    print(acc)
    return acc


def recognition(arg):
    gpu_id = 0
    set = ['0', '1', '2', '3', '4']
    view = ['55', '65', '75', '85']
    res = [arg]
    angle_sum = 0.0
    print('Res:' + str(res[0]))
    for i in view:
        print(i)
        sum = 0.0
        for j in set:
            for k in res:
                g1_path = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + i + '/SFDEI/128/Dt1/*'
                p1_path = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + i + '/SFDEI/' + k + '/Dt1/*'
                g2_path = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + i + '/SFDEI/128/Dt2/*'
                p2_path = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + i + '/SFDEI/' + k + '/Dt2/*'

                # p1_path = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_55/SFDEI/SR_exam/SRResNet8_32/resized/Dt1/*'
                # p2_path = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_55/SFDEI/SR_exam/SRResNet8_32/resized/Dt2/*'
                # p1_path = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + i + '/SFDEI/2step_downsample/Dt1/*'
                # p2_path = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + i + '/SFDEI/2step_downsample/Dt2/*'

                # path_to_g16 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/SFDEI/SR_exam/SRResNet16_64/resized/Dt' + frame + '/*'
                # path_to_p16 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/SFDEI/SR_exam/SRResNet16_64/resized/Dt' + frame + '/*'
                # path_to_g8 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/SFDEI/SR_exam/SRResNet8_32/resized/Dt' + frame + '/*'
                # path_to_p8 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/SFDEI/SR_exam/SRResNet8_32/resized/Dt' + frame + '/*'

                # テストデータのロード
                gallery1, probe1, id_list, dim = n_fold_cross_validation(n_fold=5, pare=i+'-'+i, g_path=g1_path, p_path=p1_path, test_set=j, mode='test')
                gallery2, probe2, id_list, dim = n_fold_cross_validation(n_fold=5, pare=i + '-' + i, g_path=g2_path, p_path=p2_path, test_set=j, mode='test')
                # g_test = gallery[0]
                # p_test = probe[0]
                # p_test = gallery[0]

                # modelの定義とロード
                model = two_times_two_in()
                serializers.load_npz('/home/common-ns/setoguchi/chainer_files/Two_by_two_in/SRGAN1-2/' + i + '/set_' +j+ '/model_snapshot_50000', model)
                model.to_gpu(gpu_id)

                # 識別率と不正解のid郡を返す
                # rate, uncorrect = extract_features(model, gallery1, gallery2, probe1, probe2, id_list)

                # print '識別率：{}'.format(rate)
                # print uncorrect
                # sum += rate
                acc = extract_features(model, gallery1, gallery2, probe1, probe2, id_list, gpu_id)
            sum += acc

        print('Average:' + str(sum / 5.0))
        angle_sum += sum / 5.0
    print('Angle_Mean:' + str(angle_sum / 4.0))


def usf_test(path_txt, view, subset, gpu=0):

    gpu_id = 0
    path_list = []
    with open(path_txt, 'r') as f:
        for i in f.readlines():
            path_list.append(i.replace('\n', ''))

    g1_path = sorted(glob.glob(path_list[0] + '/*'))
    g2_path = sorted(glob.glob(path_list[1] + '/*'))
    p1_path = sorted(glob.glob(path_list[2] + '/*'))
    p2_path = sorted(glob.glob(path_list[3] + '/*'))

    # idの作成
    id_list = []
    for i in range(len(g1_path)):
        id_list.append(np.int32(i))
    id_list = np.asarray(id_list)

    # バッチ化されたVariableデータセット
    if gpu == 1:
        g1 = Variable(cuda.to_gpu(load_usf(g1_path), 0))
        g2 = Variable(cuda.to_gpu(load_usf(g2_path), 0))
        p1 = Variable(cuda.to_gpu(load_usf(p1_path), 0))
        p2 = Variable(cuda.to_gpu(load_usf(p2_path), 0))
    else:
        g1 = Variable(load_usf(g1_path))
        g2 = Variable(load_usf(g2_path))
        p1 = Variable(load_usf(p1_path))
        p2 = Variable(load_usf(p2_path))

    # modelの定義とロード
    model = two_times_two_in()
    serializers.load_npz(
        os.getcwd() + '/trained_model/' + view + '/set_' + subset + '/model_snapshot_50000',
        model)

    if gpu == 1:
        model.to_gpu(0)
    p, n = model(g1, g2, p1, p2)

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(p.data, id_list)

    correct = 0.0
    for i, item in enumerate(n.data):
        predit = neigh.predict([item])[0]
        if predit == id_list[i]:
            correct = correct + 1

    acc = correct / len(n)

    print(acc)


def extract_feature_pair(path_txt, view, subset, gpu=0):

    path_list = []
    with open(path_txt, 'r') as f:
        for i in f.readlines():
            path_list.append(i.replace('\n', ''))

    q1_path = sorted(glob.glob(path_list[0] + '/*'))
    q2_path = sorted(glob.glob(path_list[1] + '/*'))

    # ファイル名の取得
    name_list = []
    for i in q1_path:
        name_list.append(os.path.basename(i).replace('.png', ''))

    # バッチ化されたVariableデータセット
    if gpu == 1:
        q1 = Variable(cuda.to_gpu(load_usf(q1_path), 0))
        q2 = Variable(cuda.to_gpu(load_usf(q2_path), 0))
    else:
        q1 = Variable(load_usf(q1_path))
        q2 = Variable(load_usf(q2_path))

    # modelの定義とロード
    model = two_times_two_in()
    serializers.load_npz(
        os.getcwd() + '/trained_model/' + view + '/set_' + subset + '/model_snapshot_50000',
        model)

    if gpu == 1:
        model.to_gpu(0)
    output = model(q1, q2)

    for i, (item, name) in enumerate(zip(output.data, name_list)):
        if i == 0:
            print('保存先として作成するディレクトリ名を入力してください')
            dir_name = input().replace('\n', '')
            # dir_name = 'tmp'
            os.mkdir(os.getcwd() + '/' + dir_name)
            np.save(os.getcwd() + '/' + dir_name + '/' + name + '.npy', item)
        else:
            np.save(os.getcwd() + '/' + dir_name + '/' + name + '.npy', item)

    '''
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(p.data, id_list)

    correct = 0.0
    for i, item in enumerate(n.data):
        predit = neigh.predict([item])[0]
        if predit == id_list[i]:
            correct = correct + 1

    acc = correct / len(n)

    print(acc)
    '''



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('text')
    parser.add_argument('view')
    parser.add_argument('subset')
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()

    # usf_test(args.text, args.view, args.subset, args.subset)
    extract_feature_pair(args.text, args.view, args.subset, args.subset)
   #  res = ['96', '64', '32', '16', '8']
   #  for i in res:
   #      recognition(i)