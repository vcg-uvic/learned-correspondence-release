# test.py ---
#
# Filename: test.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Tue Apr  3 14:17:51 2018 (-0700)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#        Move most functions to geom.py
#
#
# Copyright (C)
# Visual Computing Group @ University of Victoria
# Computer Vision Lab @ EPFL

# Code:

import datetime
import os
import sys
import time
from tqdm import tqdm
import numpy as np
from parse import parse
import cv2
from six.moves import xrange

from .utils import loadh5, saveh5
from .geom import *

def dump_val_res(#img1, img2, 
                 x1, x2, mask_before, mask_after, cx1, cy1, f1,
                 cx2, cy2, f2, R, t, dump):

    if not os.path.exists(dump):
        os.makedirs(dump)

    # Images
    #img1 = img1.transpose(1, 2, 0)
    #img2 = img2.transpose(1, 2, 0)
    #cv2.imwrite(os.path.join(dump, "img1.png"), img1)
    #cv2.imwrite(os.path.join(dump, "img2.png"), img2)

    dump_dict = {}
    dump_dict["x1"] = x1
    dump_dict["cx1"] = cx1
    dump_dict["cy1"] = cy1
    dump_dict["f1"] = f1
    dump_dict["x2"] = x2
    dump_dict["cx2"] = cx2
    dump_dict["cy2"] = cy2
    dump_dict["f2"] = f2
    dump_dict["R"] = R
    dump_dict["t"] = t

    if mask_before is not None:
        dump_dict["mask_before"] = mask_before
    if mask_after is not None:
        dump_dict["mask_after"] = mask_after

    saveh5(dump_dict, os.path.join(dump, "dump.h5"))


def test_process(mode, sess,
                 cur_global_step, merged_summary_op, summary_writer,
                 x, y, R, t, is_training,
                 img1, img2, r,
                 logits_mean, e_hat, loss,
                 data,
                 res_dir, config, va_res_only=False):

    import tensorflow as tf

    time_us = []
    time_ransac_us = []
    time_ransac = []

    inlier_us = []
    inlier_ransac = []
    inlier_ransac_us = []

    mask_dict = {}
    if mode == "test":
        print("[{}] {}: Start testing".format(config.data_te, time.asctime()))

    # Unpack some references
    xs = data["xs"]
    ys = data["ys"]
    Rs = data["Rs"]
    ts = data["ts"]
    #img1s = data["img1s"]
    cx1s = data["cx1s"]
    cy1s = data["cy1s"]
    f1s = data["f1s"]
    #img2s = data["img2s"]
    cx2s = data["cx2s"]
    cy2s = data["cy2s"]
    f2s = data["f2s"]

    # Validation
    num_sample = len(xs)

    test_list = []
    if va_res_only:
        test_list += [
            "ours",
        ]
    else:
        test_list += [
            # "ours_ransac",
            # "ours_usac5point",
            # "ours_usac8point",
            # "ours_usacnolo5point",
            # "ours_usacnolo8point",
            # "ours_ransac_weighted",
            # "ours_8point",
            # "ours_top64_ransac",
            # "ours_prob_ransac",
            # "ours_prob_ransac_weighted",
            # "ours_prob_mlesac",
            # "ours_prob_mlesac_weighted",
        ]

    eval_res = {}
    measure_list = ["err_q", "err_t", "num"]
    for measure in measure_list:
        eval_res[measure] = {}
        for _test in test_list:
            eval_res[measure][_test] = np.zeros(num_sample)

    e_hats = []
    y_hats = []
    # Run every test independently. might have different number of keypoints
    for idx_cur in tqdm(xrange(num_sample)):
        # Use minimum kp in batch to construct the batch
        _xs = np.array(
            xs[idx_cur][:, :, :]
        ).reshape(1, 1, -1, 4)
        _ys = np.array(
            ys[idx_cur][:, :]
        ).reshape(1, -1, 2)
        _dR = np.array(Rs[idx_cur]).reshape(1, 9)
        _dt = np.array(ts[idx_cur]).reshape(1, 3)
        # Create random permutation indices
        feed_dict = {
            x: _xs,
            y: _ys,
            R: _dR,
            t: _dt,
            is_training:  config.net_bn_test_is_training,
        }
        fetch = {
            "e_hat": e_hat,
            "y_hat": logits_mean,
            "loss": loss,
            # "summary": merged_summary_op,
            # "global_step": global_step,
        }
        # print("Running network for {} correspondences".format(
        #     _xs.shape[2]
        # ))
        time_start = datetime.datetime.now()
        res = sess.run(fetch, feed_dict=feed_dict)
        time_end = datetime.datetime.now()
        time_diff = time_end - time_start
        # print("Runtime in milliseconds: {}".format(
        #     float(time_diff.total_seconds() * 1000.0)
        # ))
        time_us += [time_diff.total_seconds() * 1000.0]

        e_hats.append(res["e_hat"])
        y_hats.append(res["y_hat"])

    for cur_val_idx in tqdm(xrange(num_sample)):
        _xs = xs[cur_val_idx][:, :, :].reshape(1, 1, -1, 4)
        _ys = ys[cur_val_idx][:, :].reshape(1, -1, 2)
        _dR = Rs[cur_val_idx]
        _dt = ts[cur_val_idx]
        e_hat_out = e_hats[cur_val_idx].flatten()
        y_hat_out = y_hats[cur_val_idx].flatten()
        if len(y_hat_out) != _xs.shape[2]:
            y_hat_out = np.ones(_xs.shape[2])
        # Eval decompose for all pairs
        _xs = _xs.reshape(-1, 4)
        # x coordinates
        _x1 = _xs[:, :2]
        _x2 = _xs[:, 2:]
        # current validity from network
        _valid = y_hat_out.flatten()
        # choose top ones (get validity threshold)
        _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
        _relu_tanh = np.maximum(0, np.tanh(_valid))

        
        _mask_before = _valid >= max(0, _valid_th)
        mask_dict[cur_val_idx] = _mask_before
        # For every things to test
        _use_prob = True
        for _test in test_list:
            if _test == "ours":
                _eval_func = "non-decompose"
                _mask_before = _valid >= max(0, _valid_th)
                _method = None
                _probs = None
                _weighted = False
            elif _test == "ours_ransac":
                _eval_func = "decompose"
                _mask_before = _valid >= max(0, _valid_th)
                _method = cv2.RANSAC
                _probs = None
                _weighted = False

            if _eval_func == "non-decompose":
                _err_q, _err_t, _, _, _num_inlier, \
                    _ = eval_nondecompose(
                        _x1, _x2, e_hat_out, _dR, _dt, y_hat_out)
                _mask_after = _mask_before
            elif _eval_func == "decompose":
                # print("RANSAC loop with ours")
                time_start = datetime.datetime.now()
                _err_q, _err_t, _, _, _num_inlier, \
                    _mask_after = eval_decompose(
                        _x1, _x2, _dR, _dt, mask=_mask_before,
                        method=_method, probs=_probs,
                        weighted=_weighted, use_prob=_use_prob)
                time_end = datetime.datetime.now()
                time_diff = time_end - time_start
                # print("Runtime in milliseconds: {}".format(
                #     float(time_diff.total_seconds() * 1000.0)
                # ))
                # print("RANSAC loop without ours")
                inlier_us += [np.sum(_mask_before)]
                inlier_ransac_us += [np.sum(_mask_after)]
                time_ransac_us += [time_diff.total_seconds() * 1000.0]
                time_start = datetime.datetime.now()
                _, _, _, _, _, \
                    _mask_tmp = eval_decompose(
                        _x1, _x2, _dR, _dt,
                        mask=np.ones_like(_mask_before).astype(bool),
                        method=_method, probs=_probs,
                        weighted=_weighted, use_prob=_use_prob)
                time_end = datetime.datetime.now()
                time_diff = time_end - time_start
                inlier_ransac += [np.sum(_mask_tmp)]
                # print("Runtime in milliseconds: {}".format(
                #     float(time_diff.total_seconds() * 1000.0)
                # ))
                time_ransac += [time_diff.total_seconds() * 1000.0]

            # Load them in list
            eval_res["err_q"][_test][cur_val_idx] = _err_q
            eval_res["err_t"][_test][cur_val_idx] = _err_t
            eval_res["num"][_test][cur_val_idx] = _num_inlier

            
            if config.vis_dump:
                dump_val_res(
                    #img1s[cur_val_idx],
                    #img2s[cur_val_idx],
                    _x1, _x2, _mask_before, _mask_after,
                    cx1s[cur_val_idx],
                    cy1s[cur_val_idx],
                    f1s[cur_val_idx],
                    cx2s[cur_val_idx],
                    cy2s[cur_val_idx],
                    f2s[cur_val_idx],
                    Rs[cur_val_idx],
                    ts[cur_val_idx],
                    os.path.join(
                        res_dir, mode, "match", _test,
                        "pair{:08d}".format(cur_val_idx)
                    ),
                )

        # print("Test {}".format(_test))
        # print("Time taken to compute us {}".format(np.median(time_us)))
        # print("Time taken to compute ransac {}".format(np.median(time_ransac)))
        # print("Time taken to compute ransac after us {}".format(
        #     np.median(time_ransac_us)))
        # print("Inliers us {}".format(np.median(inlier_us)))
        # print("Inliers ransac {}".format(np.median(inlier_ransac)))
        # print("Inliers ransac + us {}".format(np.median(inlier_ransac_us)))

    if config.vis_dump:
        print("[{}] {}: End dumping".format(
            config.data_tr, time.asctime()))
        assert config.run_mode != "train"
        return np.nan

    summaries = []
    ret_val = 0
    for _tag in test_list:
        for _sub_tag in measure_list:
            summaries.append(
                tf.Summary.Value(
                    tag="ErrorComputation/" + _tag,
                    simple_value=np.median(eval_res[_sub_tag][_tag])
                )
            )

            # For median error
            ofn = os.path.join(
                res_dir, mode, "median_{}_{}.txt".format(_sub_tag, _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(
                    np.median(eval_res[_sub_tag][_tag])))

        ths = np.arange(7) * 5
        cur_err_q = np.array(eval_res["err_q"][_tag]) * 180.0 / np.pi
        cur_err_t = np.array(eval_res["err_t"][_tag]) * 180.0 / np.pi
        # Get histogram
        q_acc_hist, _ = np.histogram(cur_err_q, ths)
        t_acc_hist, _ = np.histogram(cur_err_t, ths)
        qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
        num_pair = float(len(cur_err_q))
        q_acc_hist = q_acc_hist.astype(float) / num_pair
        t_acc_hist = t_acc_hist.astype(float) / num_pair
        qt_acc_hist = qt_acc_hist.astype(float) / num_pair
        q_acc = np.cumsum(q_acc_hist)
        t_acc = np.cumsum(t_acc_hist)
        qt_acc = np.cumsum(qt_acc_hist)
        # Store return val
        if _tag == "ours":
            ret_val = np.mean(qt_acc[:4])  # 1 == 5
        for _idx_th in xrange(1, len(ths)):
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_q_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(q_acc[:_idx_th]),
                )
            ]
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_t_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(t_acc[:_idx_th]),
                )
            ]
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_qt_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(qt_acc[:_idx_th]),
                )
            ]
            # for q_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_q_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(q_acc[:_idx_th])))
            # for qt_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_t_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(t_acc[:_idx_th])))
            # for qt_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_qt_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(qt_acc[:_idx_th])))

    # summary_writer.add_summary(
    #     tf.Summary(value=summaries), global_step=cur_global_step)

    if mode == "test":
        print("[{}] {}: End testing".format(
            config.data_tr, time.asctime()))

    # Return qt_auc20 of ours
    return ret_val, mask_dict


def comp_process(mode, data, res_dir, config):
    import parse

    # Unpack some references
    xs = data["xs"]
    ys = data["ys"]
    Rs = data["Rs"]
    ts = data["ts"]
    img1s = data["img1s"]
    cx1s = data["cx1s"]
    cy1s = data["cy1s"]
    f1s = data["f1s"]
    img2s = data["img2s"]
    cx2s = data["cx2s"]
    cy2s = data["cy2s"]
    f2s = data["f2s"]

    # Make fs numpy array
    f1s = np.array(f1s)
    f2s = np.array(f2s)

    # Prepare directory
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print("[{}] {}: Start testing".format(
        config.data_tr, time.asctime()))
    # Validation
    num_sample = len(xs)

    if config.use_lift:
        comp_list = [
            "lmeds", "ransac", "mlesac",
            # "usac5point", "usac8point",
            # "usacnolo5point", "usacnolo8point"
        ]
    else:
        # comp_list = [
        #     "lmeds", "ransac", "top8_8point", "top50_lmeds", "top50_ransac",
        #     "gms", "gms_orb", "gms_default",
        #     "gms_orb_resize", "gms_orb_resize_ransac",
        #     "gms_orb_resize_tt", "gms_orb_resize_ransac_tt",
        #     "mlesac",
        # ]
        comp_list = [
            "lmeds", "ransac", "mlesac",
            # "usac5point", "usac8point",
            # "usacnolo5point", "usacnolo8point"
        ]
        if config.obj_num_kp == 2000:
            comp_list += [
                "gms_orb_resize_ransac_tt",
                "gms_orb_resize_tt",
            ]

    # Initialize arrays that will store measurements
    err_q = {}
    err_t = {}
    num = {}
    for _comp in comp_list:
        err_q[_comp] = np.zeros(num_sample)
        err_t[_comp] = np.zeros(num_sample)
        num[_comp] = np.zeros(num_sample)

    NUM_KP = config.obj_num_kp
    # batch_size = config.val_batch_size
    # num_batch = int(len(xs) / batch_size)

    from gms_matcher import GmsMatcher

    # SIFT
    sift = cv2.xfeatures2d.SIFT_create(
        nfeatures=NUM_KP, contrastThreshold=1e-5)
    if cv2.__version__.startswith('3'):
        sift_matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        sift_matcher = cv2.BFMatcher_create(cv2.NORM_L2)
    sift_gms = GmsMatcher(sift, sift_matcher)

    # ORB
    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)
    if cv2.__version__.startswith('3'):
        orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        orb_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    orb_gms = GmsMatcher(orb, orb_matcher)

    for method_name in comp_list:

        # Check res_dir if we have the dump ready
        dump_dir = os.path.join(res_dir, mode, method_name)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        # Check for full dump
        full_dump_file = os.path.join(dump_dir, "qtn_all.h5")
        if not os.path.exists(full_dump_file) or config.vis_dump:
            for _idx in xrange(num_sample):
                print("\rWorking on {} / {}".format(
                    _idx + 1, num_sample), end="")
                sys.stdout.flush()
                dump_file = os.path.join(dump_dir, "qtn_{}.txt".format(_idx))
                # If dump exists, just load it
                if os.path.exists(dump_file) and not config.vis_dump:
                    with open(dump_file, "r") as ifp:
                        dump_res = ifp.read()
                    dump_res = parse(
                        "{err_q:e}, {err_t:e}, {num_inlier:d}\n", dump_res)
                    _err_q = dump_res["err_q"]
                    _err_t = dump_res["err_t"]
                    _num_inlier = dump_res["num_inlier"]
                else:
                    # Otherwise compute
                    _xs = xs[_idx][:, :, :].reshape(1, 1, -1, 4)
                    _ys = ys[_idx][:, :].reshape(1, -1, 2)
                    _dR = Rs[_idx]
                    _dt = ts[_idx]
                    # Eval decompose for all pairs
                    _xs = _xs.reshape(-1, 4)
                    # x coordinates
                    _x1 = _xs[:, :2]
                    _x2 = _xs[:, 2:]

                    # Prepare input
                    if method_name == "lmeds":
                        eval_func = eval_decompose
                        _method = cv2.LMEDS
                        _mask = None
                    elif method_name == "ransac":
                        eval_func = eval_decompose
                        _method = cv2.RANSAC
                        _mask = None
                    elif method_name == "mlesac":
                        eval_func = eval_decompose
                        _method = "MLESAC"
                        _mask = None
                    elif method_name == "gms":
                        eval_func = eval_decompose_8points
                        _method = None
                        sift_gms.empty_matches()
                        _x1, _x2, _mask = sift_gms.compute_matches(
                            np.transpose(img1s[_idx], (1, 2, 0)),
                            np.transpose(img2s[_idx], (1, 2, 0)),
                            cx1s[_idx], cx2s[_idx],
                            cy1s[_idx], cy2s[_idx],
                            f1s[_idx], f2s[_idx],
                            with_scale=True, with_rotation=True
                        )
                    elif method_name == "gms_default":
                        eval_func = eval_decompose_8points
                        _method = None
                        orb_gms.empty_matches()
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            np.transpose(img1s[_idx], (1, 2, 0)),
                            np.transpose(img2s[_idx], (1, 2, 0)),
                            cx1s[_idx], cx2s[_idx],
                            cy1s[_idx], cy2s[_idx],
                            f1s[_idx], f2s[_idx],
                            with_scale=False, with_rotation=False
                        )
                    elif method_name == "gms_orb_resize_ransac_tt":
                        eval_func = eval_decompose
                        _method = cv2.RANSAC
                        orb_gms.empty_matches()
                        _img1 = np.transpose(img1s[_idx], (1, 2, 0))
                        _img2 = np.transpose(img2s[_idx], (1, 2, 0))
                        _h1, _w1 = _img1.shape[:2]
                        _h2, _w2 = _img2.shape[:2]
                        _s1 = 480.0 / _h1
                        _s2 = 480.0 / _h2
                        _h1 = int(_h1 * _s1)
                        _w1 = int(np.round(_w1 * _s1))
                        _h2 = int(_h2 * _s2)
                        _w2 = int(np.round(_w2 * _s2))
                        _img1 = cv2.resize(_img1, (_w1, _h1))
                        _img2 = cv2.resize(_img2, (_w2, _h2))
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            _img1, _img2,
                            cx1s[_idx] * _s1, cx2s[_idx] * _s2,
                            cy1s[_idx] * _s1, cy2s[_idx] * _s2,
                            f1s[_idx] * _s1, f2s[_idx] * _s2,
                            with_scale=True, with_rotation=True
                        )
                    elif method_name == "gms_orb_resize_tt":
                        eval_func = eval_decompose_8points
                        _method = None
                        orb_gms.empty_matches()
                        _img1 = np.transpose(img1s[_idx], (1, 2, 0))
                        _img2 = np.transpose(img2s[_idx], (1, 2, 0))
                        _h1, _w1 = _img1.shape[:2]
                        _h2, _w2 = _img2.shape[:2]
                        _s1 = 480.0 / _h1
                        _s2 = 480.0 / _h2
                        _h1 = int(_h1 * _s1)
                        _w1 = int(np.round(_w1 * _s1))
                        _h2 = int(_h2 * _s2)
                        _w2 = int(np.round(_w2 * _s2))
                        _img1 = cv2.resize(_img1, (_w1, _h1))
                        _img2 = cv2.resize(_img2, (_w2, _h2))
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            _img1, _img2,
                            cx1s[_idx] * _s1, cx2s[_idx] * _s2,
                            cy1s[_idx] * _s1, cy2s[_idx] * _s2,
                            f1s[_idx] * _s1, f2s[_idx] * _s2,
                            with_scale=True, with_rotation=True
                        )
                    elif method_name == "gms_orb_resize_ransac":
                        eval_func = eval_decompose
                        _method = cv2.RANSAC
                        orb_gms.empty_matches()
                        _img1 = np.transpose(img1s[_idx], (1, 2, 0))
                        _img2 = np.transpose(img2s[_idx], (1, 2, 0))
                        _h1, _w1 = _img1.shape[:2]
                        _h2, _w2 = _img2.shape[:2]
                        _s1 = 480.0 / _h1
                        _s2 = 480.0 / _h2
                        _h1 = int(_h1 * _s1)
                        _w1 = int(np.round(_w1 * _s1))
                        _h2 = int(_h2 * _s2)
                        _w2 = int(np.round(_w2 * _s2))
                        _img1 = cv2.resize(_img1, (_w1, _h1))
                        _img2 = cv2.resize(_img2, (_w2, _h2))
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            _img1, _img2,
                            cx1s[_idx] * _s1, cx2s[_idx] * _s2,
                            cy1s[_idx] * _s1, cy2s[_idx] * _s2,
                            f1s[_idx] * _s1, f2s[_idx] * _s2,
                            with_scale=False, with_rotation=False
                        )
                    elif method_name == "gms_orb_resize":
                        eval_func = eval_decompose_8points
                        _method = None
                        orb_gms.empty_matches()
                        _img1 = np.transpose(img1s[_idx], (1, 2, 0))
                        _img2 = np.transpose(img2s[_idx], (1, 2, 0))
                        _h1, _w1 = _img1.shape[:2]
                        _h2, _w2 = _img2.shape[:2]
                        _s1 = 480.0 / _h1
                        _s2 = 480.0 / _h2
                        _h1 = int(_h1 * _s1)
                        _w1 = int(np.round(_w1 * _s1))
                        _h2 = int(_h2 * _s2)
                        _w2 = int(np.round(_w2 * _s2))
                        _img1 = cv2.resize(_img1, (_w1, _h1))
                        _img2 = cv2.resize(_img2, (_w2, _h2))
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            _img1, _img2,
                            cx1s[_idx] * _s1, cx2s[_idx] * _s2,
                            cy1s[_idx] * _s1, cy2s[_idx] * _s2,
                            f1s[_idx] * _s1, f2s[_idx] * _s2,
                            with_scale=False, with_rotation=False
                        )
                    elif method_name == "gms_orb":
                        eval_func = eval_decompose_8points
                        _method = None
                        orb_gms.empty_matches()
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            np.transpose(img1s[_idx], (1, 2, 0)),
                            np.transpose(img2s[_idx], (1, 2, 0)),
                            cx1s[_idx], cx2s[_idx],
                            cy1s[_idx], cy2s[_idx],
                            f1s[_idx], f2s[_idx],
                            with_scale=True, with_rotation=True
                        )

                    # Compute errors
                    _err_q, _err_t, _, _, _num_inlier, _mask_after = eval_func(
                        _x1, _x2, _dR, _dt, mask=_mask, method=_method)

                    if config.vis_dump:
                        dump_val_res(
                            #img1s[_idx],
                            #img2s[_idx],
                            _x1, _x2, _mask, _mask_after,
                            cx1s[_idx],
                            cy1s[_idx],
                            f1s[_idx],
                            cx2s[_idx],
                            cy2s[_idx],
                            f2s[_idx],
                            Rs[_idx],
                            ts[_idx],
                            os.path.join(
                                res_dir, mode, "match", method_name,
                                "pair{:08d}".format(_idx)
                            ),
                        )
                    else:
                        # Write dump
                        with open(dump_file, "w") as ofp:
                            ofp.write("{:e}, {:e}, {:d}\n".format(
                                _err_q, _err_t, _num_inlier))

                # Load them in list
                err_q[method_name][_idx] = _err_q
                err_t[method_name][_idx] = _err_t
                num[method_name][_idx] = _num_inlier

            if not config.vis_dump:
                # Save to full dump
                dump_dict = {}
                dump_dict["err_q"] = err_q[method_name]
                dump_dict["err_t"] = err_t[method_name]
                dump_dict["num"] = num[method_name]
                saveh5(dump_dict, full_dump_file)

            # Remove all intermediate cache
            for _f in os.listdir(dump_dir):
                if _f.startswith("qtn_") and _f.endswith(".txt"):
                    os.remove(os.path.join(dump_dir, _f))

        # Load the full dump file
        else:
            dump_dict = loadh5(full_dump_file)
            err_q[method_name] = dump_dict["err_q"]
            err_t[method_name] = dump_dict["err_t"]
            num[method_name] = dump_dict["num"]

            # Remove all intermediate cache
            for _f in os.listdir(dump_dir):
                if _f.startswith("qtn_") and _f.endswith(".txt"):
                    os.remove(os.path.join(dump_dir, _f))

    print("")

    if config.vis_dump:
        return

    # Report results
    for _tag in comp_list:
        # For median error
        ofn = os.path.join(res_dir, mode, "median_err_q_{}.txt".format(_tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.median(err_q[_tag])))
        ofn = os.path.join(res_dir, mode, "median_err_t_{}.txt".format(_tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.median(err_t[_tag])))
        ofn = os.path.join(res_dir, mode, "median_num_{}.txt".format(_tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.median(num[_tag])))

        # For accuracy AUC
        ths = np.arange(7) * 5
        cur_err_q = np.array(err_q[_tag]) * 180.0 / np.pi
        cur_err_t = np.array(err_t[_tag]) * 180.0 / np.pi
        # Get histogram
        q_acc_hist, _ = np.histogram(cur_err_q, ths)
        t_acc_hist, _ = np.histogram(cur_err_t, ths)
        qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
        num_pair = float(len(cur_err_q))
        q_acc_hist = q_acc_hist.astype(float) / num_pair
        t_acc_hist = t_acc_hist.astype(float) / num_pair
        qt_acc_hist = qt_acc_hist.astype(float) / num_pair
        q_acc = np.cumsum(q_acc_hist)
        t_acc = np.cumsum(t_acc_hist)
        qt_acc = np.cumsum(qt_acc_hist)
        for _idx_th in xrange(1, len(ths)):
            # for q_auc
            ofn = os.path.join(
                res_dir, mode, "acc_q_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(q_acc[:_idx_th])))
            # for t_auc
            ofn = os.path.join(
                res_dir, mode, "acc_t_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(t_acc[:_idx_th])))
            # for qt_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_qt_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(qt_acc[:_idx_th])))

    print("[{}] {}: End testing".format(config.data_tr, time.asctime()))


#
# test.py ends here
