# _*_ coding:utf-8 _*_

import os
import math
import chainer
import glob
import numpy as np
from PIL import Image


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


def split_dataset(gallery_data, probe_data, n_fold, test_set, mode):

    num_of_subset = int(math.ceil(len(gallery_data) / float(n_fold)))
    idx = 0
    # 全データをN分割する
    for j in range(n_fold):
        if j == 0:
            g_split = [gallery_data[idx:idx + num_of_subset]]
            p_split = [probe_data[idx:idx + num_of_subset]]
            idx += num_of_subset
        else:
            g_split.append(gallery_data[idx:idx + num_of_subset])
            p_split.append(probe_data[idx:idx + num_of_subset])
            idx += num_of_subset

    # testに使うデータのみ抽出，その後リストから削除
    g_test = g_split[int(test_set)]
    p_test = p_split[int(test_set)]
    del g_split[int(test_set)]
    del p_split[int(test_set)]

    # 分割したデータをフラットに戻して，train setを作成
    g_train = flatten_with_any_depth(g_split)
    p_train = flatten_with_any_depth(p_split)

    # ラベルを作成
    id_list = []
    for i in range(len(g_train)):
        id_list.append(np.int32(i))

    if mode == 'train':
        return g_train, p_train, len(g_train)
    elif mode == 'test':
        return g_test, p_test, len(g_train)


# N分割したデータをロード
def n_fold_cross_validation(n_fold, pare, g_path, p_path, test_set, mode='train'):
    # GalleryとProbeの各データをロード
    path_to_galleries = sorted(glob.glob(g_path))
    path_to_probes = sorted(glob.glob(p_path))

    g_list = []
    p_list = []
    id_list = []
    common_id_list = []
    num = 0
    # 観測方向によって含まれる人物が異なるため，名簿リストを参照
    txt = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/common_id_' + pare + '.txt'
    with open(txt, 'r') as f:
        common_id = f.readlines()
        for i, (gallery, probe) in enumerate(zip(path_to_galleries, path_to_probes)):

            id = os.path.basename(gallery.replace('.png', '\n'))
            if id not in common_id:
                num += 1
                continue

            # PILからnumpyに型変換
            g_item = np.asarray(Image.open(gallery))
            g_array = g_item.astype(np.float32) / 255.0

            if g_array.size == 11264:
                g_list.append(g_array.reshape((1, 128, 88)))
            elif g_array.size == 33792:
                g_list.append(g_array.transpose([2, 0, 1]))

            p_item = np.asarray(Image.open(probe))
            p_array = p_item.astype(np.float32) / 255.0

            if g_array.size == 11264:
                p_list.append(p_array.reshape((1, 128, 88)))
            elif p_array.size == 33792:
                p_list.append(p_array.transpose([2, 0, 1]))

            if mode == 'common':
                common_id_list.append(id.replace('\n', ''))
                # common_id_list.append(i)

    if mode == 'common':
        return g_list, p_list, common_id_list

    g, p, size_fc4 = split_dataset(g_list, p_list, n_fold=n_fold, test_set=test_set, mode=mode)

    for i in range(len(g)):
        id_list.append(np.int32(i))

    return g, p, id_list, size_fc4


# 学習データのロード
def load_res_invariant_dataset(view, set, type='GEI', frame=None):
    if type=='GEI':
        path_to_g128 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/GEI/128/*'
        path_to_p128 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/GEI/128/*'
        path_to_g96 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/GEI/96/*'
        path_to_p96 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/GEI/96/*'
        path_to_g64 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/GEI/64/*'
        path_to_p64 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/GEI/64/*'
        path_to_g32 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/GEI/32/*'
        path_to_p32 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/GEI/32/*'
        path_to_g16 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/GEI/16/*'
        path_to_p16 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/GEI/16/*'
        path_to_g8 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/GEI/8/*'
        path_to_p8 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/GEI/8/*'
        # path_to_g4 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/GEI/4/*'
        # path_to_p4 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/GEI/4/*'
    elif type=='SFDEI':
        path_to_g128 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/SFDEI/128/Dt' + frame + '/*'
        path_to_p128 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/SFDEI/128/Dt' + frame + '/*'
        path_to_g96 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/SFDEI/96/Dt' + frame + '/*'
        path_to_p96 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/SFDEI/96/Dt' + frame + '/*'
        path_to_g64 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/SFDEI/64/Dt' + frame + '/*'
        path_to_p64 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/SFDEI/64/Dt' + frame + '/*'
        path_to_g32 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/SFDEI/32/Dt' + frame + '/*'
        path_to_p32 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/SFDEI/32/Dt' + frame + '/*'
        # path_to_g16 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/SFDEI/16/Dt' + frame + '/*'
        # path_to_p16 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/SFDEI/16/Dt' + frame + '/*'
        # path_to_g8 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_' + view + '/SFDEI/8/Dt' + frame + '/*'
        # path_to_p8 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_' + view + '/SFDEI/8/Dt' + frame + '/*'
        path_to_g16 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_'+ view +'/SFDEI/SR_exam/SRGAN16_64/resized/Dt' + frame + '/*'
        path_to_p16 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_'+ view +'/SFDEI/SR_exam/SRGAN16_64/resized/Dt' + frame + '/*'
        path_to_g8 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Gallery_'+ view +'/SFDEI/SR_exam/SRGAN8_32/resized/Dt' + frame + '/*'
        path_to_p8 = '/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/5_fold_cross_validation/subset_A/Bilinear_Probe_'+ view +'/SFDEI/SR_exam/SRGAN8_32/resized/Dt' + frame + '/*'

    # 各解像度のデータをロード
    g128, p128, id_128 = n_fold_cross_validation(n_fold=5, pare=view+'-'+view, g_path=path_to_g128, p_path=path_to_p128,
                                                mode='common', test_set=set)
    g96, p96, id_96 = n_fold_cross_validation(n_fold=5, pare=view+'-'+view, g_path=path_to_g96, p_path=path_to_p96,
                                                mode='common', test_set=set)
    g64, p64, id_64 = n_fold_cross_validation(n_fold=5, pare=view+'-'+view, g_path=path_to_g64, p_path=path_to_p64,
                                                mode='common', test_set=set)
    g32, p32, id_32 = n_fold_cross_validation(n_fold=5, pare=view+'-'+view, g_path=path_to_g32, p_path=path_to_p32,
                                                mode='common', test_set=set)
    g16, p16, id_16 = n_fold_cross_validation(n_fold=5, pare=view+'-'+view, g_path=path_to_g16, p_path=path_to_p16,
                                                mode='common', test_set=set)
    g8, p8, id_8 = n_fold_cross_validation(n_fold=5, pare=view + '-' + view, g_path=path_to_g8, p_path=path_to_p8,
                                              mode='common', test_set=set)
    # g4, p4, id_4 = n_fold_cross_validation(n_fold=5, pare=view + '-' + view, g_path=path_to_g4, p_path=path_to_p4,
    #                                           mode='common', test_set=set)

    g_common_list = []
    p_common_list = []
    all_id_list = []

    for i in range(0, len(g128)):

        g_common_list.append([g128[i], g96[i], g64[i], g32[i], g16[i], g8[i]])
        p_common_list.append([p128[i], p96[i], p64[i], p32[i], p16[i], p8[i]])
        all_id_list.append(i)

    g_common_list = chainer.datasets.TupleDataset(g_common_list, all_id_list)
    p_common_list = chainer.datasets.TupleDataset(p_common_list, all_id_list)

    g, p, _ = split_dataset(g_common_list, p_common_list, n_fold=5, test_set=set, mode='train')

    return g, p


def load_usf(dir):

    for num, i in enumerate(dir):
        if num == 0:
            img = np.asarray(Image.open(i)).transpose([2, 0, 1]) / 255.0
            images = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        else:
            img = np.asarray(Image.open(i)).transpose([2, 0, 1]) / 255.0
            images = np.vstack((images, img.reshape(1, img.shape[0], img.shape[1], img.shape[2])))

    images = images.astype(np.float32)

    return images