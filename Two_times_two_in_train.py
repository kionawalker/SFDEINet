# _*_ coding:utf-8 _*_
# written by Naoki Setoguchi
# date 27/08/2018

import chainer
from chainer import functions as F
from chainer import training, reporter, iterators, optimizers, serializers
from chainer.training import extensions
from chainer.backends import cuda
import chainer.computational_graph as c
import numpy as np
import copy
import random
import os
from tensorboardX import SummaryWriter
import sys

sys.path.append('/home/common-ns/PycharmProjects/Multiscale_SFDEINet/4inputs')
from Two_times_two_in import two_by_two_in

sys.path.append("/home/common-ns/PycharmProjects/Multiscale_SFDEINet/4inputs")
from loader import load_res_invariant_dataset


def create_pair(gallery1, gallery2, probe1, probe2):
    # バッチの約1/10を正例にする
    size = int(len(gallery1) * 0.1)
    # 正例にするインデックスをまとめたリストをつくる
    seq = np.arange(0, len(gallery1))
    id_list = np.asarray(random.sample(seq, k=size))
    rate = 0.0
    dataset = []

    for idx, (g1, g2, p1, p2) in enumerate(zip(gallery1, gallery2, probe1, probe2)):
        # id_list内の値と同じindexをもつデータは正例とする
        flag = np.where(id_list == idx, idx, -1)  # 該当する要素は正例のindex，それ以外は-1にした後，全体の配列を返す
        # flagの最大値は正例をつくるインデックスを示す
        if flag.max() > -1:

            g_item1, g_item2 = check_image(g1, g2)
            p_item1, p_item2 = check_image(p1, p2)
            item = (g_item1, g_item2, p_item1, p_item2)
            # if g[1] != p[1]:
            #     print 'error:正例がまちがっている'
            dataset.append((item, 1))
            rate += 1

        # 上記以外はprobeからランダムにペアを選ぶ
        else:

            g_item1, g_item2 = check_image(g1, g2)
            # 異なる人物のペアになるまで乱数を生成
            while True:
                # ランダム値の重複は許容
                seed = np.random.randint(0, len(gallery1))
                if seed != idx:
                    break
            p_item1, p_item2 = check_image(probe1[seed], probe2[seed])
            item = (g_item1, g_item2, p_item1, p_item2)
            # if g[1] == probe[seed][1]:
            #     print 'error:負例がまちがっている'
            dataset.append((item, 0))

    # 正例の割合を表示
    # print float(rate)/float(len(gallery))*100.0
    return dataset


def check_image(data1, data2):
    tmp1 = copy.deepcopy(data1)
    tmp2 = copy.deepcopy(data2)
    while True:
        if len(tmp1[0]) == 1:
            out1 = tmp1[0][0]
            out2 = tmp2[0][0]
            # out = out[0]
            break
        else:
            res_seed = np.random.randint(0, len(tmp1[0]))

        if tmp1[0][res_seed].max() != 0.0:
            out1 = tmp1[0][res_seed]
            out2 = tmp2[0][res_seed]
            break
        del tmp1[0][res_seed]
        del tmp2[0][res_seed]

    return out1, out2


def make_batch_list(dataset, batch_size):
    num = 0
    # ループを回す回数を決定
    times = len(dataset) / batch_size
    remainder = len(dataset) % batch_size
    # 割り切れなかったらbatch数を増やして端数を格納
    if remainder != 0:
        times += 1
    for j in range(int(times)):
        if j == 0:
            batch_dataset = [dataset[num:num + batch_size]]
            num += batch_size
        else:
            batch_dataset.append(dataset[num:num + batch_size])
            num += batch_size

    return batch_dataset


def flatten_with_any_depth(nested_list):
    """深さ優先探索の要領で入れ子のリストをフラットにする関数"""
    # フラットなリストとフリンジを用意
    flat_list = []
    fringe = [nested_list]

    while len(fringe) > 0:
        node = fringe.pop(0)
        # ノードがリストであれば子要素をフリンジに追加
        # リストでなければそのままフラットリストに追加
        if isinstance(node, list):
            fringe = node + fringe
        else:
            flat_list.append(node)

    return flat_list


def list2npy(batch, id):
    for i, data in enumerate(batch):
        pare = data[0]
        if i == 0:
            pos1 = pare[0].reshape(1, 3, 128, 88)
            pos2 = pare[1].reshape(1, 3, 128, 88)
            neg1 = pare[2].reshape(1, 3, 128, 88)
            neg2 = pare[3].reshape(1, 3, 128, 88)
            signal = data[1]
        else:
            pos1 = np.vstack((pos1, pare[0].reshape(1, 3, 128, 88)))
            pos2 = np.vstack((pos2, pare[1].reshape(1, 3, 128, 88)))
            neg1 = np.vstack((neg1, pare[2].reshape(1, 3, 128, 88)))
            neg2 = np.vstack((neg2, pare[3].reshape(1, 3, 128, 88)))
            signal = np.vstack((signal, data[1]))

    pos1 = cuda.to_gpu(pos1, id)
    pos2 = cuda.to_gpu(pos2, id)
    neg1 = cuda.to_gpu(neg1, id)
    neg2 = cuda.to_gpu(neg2, id)
    signal = chainer.Variable(cuda.to_gpu(signal, id))

    return pos1, pos2, neg1, neg2, signal


def train(view, set):
    # 5分割されたデータのロード
    gallery1, probe1 = load_res_invariant_dataset(view=view, set=set, type='SFDEI', frame='1')
    gallery2, probe2 = load_res_invariant_dataset(view=view, set=set, type='SFDEI', frame='2')

    # tensor-boardの定義
    writer = SummaryWriter()

    # 学習の設定
    save_dir = '/home/common-ns/PycharmProjects/Multiscale_SFDEINet/4inputs/models'+ view + '/set_' + str(set)
    # save_dir = '/home/common-ns/setoguchi/chainer_files/Two_by_two_in/SRGAN1-2/' + view + '/set_' + str(set)
    os.mkdir(save_dir)
    batch_size = 239
    max_iteration = 50000
    id = 1
    # gpu_id = chainer.backends.cuda.get_device_from_id(id)
    model = two_by_two_in()
    model.to_gpu(id)
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    iteration = 1.0
    i = 0
    # 学習ループ
    # for i in range(1, epock+1):
    while iteration < max_iteration + 1:

        accum_loss = 0.0
        count = 0

        # データから正例，負例のペアを構築．少なくとも全データの1割が正例になるようにする
        # データに偏りが出ないよう，データのタイプを1/4の確率で決定
        select_seed = np.random.randint(0, 2)
        if select_seed == 0:
            train_data = create_pair(gallery1, gallery2, probe1, probe2)
        elif select_seed == 1:
            train_data = create_pair(probe1, probe2, gallery1, gallery2)
        elif select_seed == 2:
            train_data = create_pair(gallery1, gallery2, gallery1, gallery2)
        elif select_seed == 3:
            train_data = create_pair(probe1, probe2, probe1, probe2)

        # バッチをつくる前にシャッフル
        shuffle = random.sample(train_data, len(train_data))
        # ミニバッチに分割
        mini_batches = make_batch_list(shuffle, batch_size)

        for batch in mini_batches:
            # リストから1つのnumpyバッチにする
            pos1, pos2, neg1, neg2, signal = list2npy(batch, id)

            # Forward
            g_out, p_out = model(pos1, pos2, neg1, neg2)

            # lossの計算
            signal = F.flatten(signal)  # contrastive_lossの仕様に合わせて1次元化
            loss = F.contrastive(g_out, p_out, signal, margin=3, reduce='mean')
            # print cuda.to_cpu(loss.data)
            accum_loss += cuda.to_cpu(loss.data)

            # backward
            model.cleargrads()
            loss.backward()

            # パラメータ更新
            optimizer.update()

            if iteration % 10000 == 0:
                optimizer.lr = optimizer.lr * 0.1
                print('学習率がさがりました：{}'.format(optimizer.lr))
            if iteration % 5000 == 0:
                serializers.save_npz(save_dir + '/model_snapshot_{}'.format(int(iteration)), model)
            iteration += 1.0
        i += 1

        print('epock:{}'.format(i), 'iteration:{}'.format(int(iteration)), 'accum_loss:{}'.format(
            accum_loss / float(len(mini_batches))))
        writer.add_scalar('train/loss', accum_loss / float(len(mini_batches)), i)

        # 約1000イテレーション毎にモデルを保存
        # if iteration % 10000 == 0:
        #     serializers.save_npz(save_dir + '/model_snapshot_{}'.format(i), model)
    g = c.build_computational_graph(g_out[0])
    with open(save_dir + '/graph.dot', 'w') as o:
        o.write(g.dump())


if __name__ == '__main__':
    subset = [0, 1, 2, 3, 4]
    view = ['55', '65']
    for i in view:
        for j in subset:
            train(i, j)
