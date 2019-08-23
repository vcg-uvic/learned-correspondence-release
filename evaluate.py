# Code based on kwang's original test.py script
# Customized test script, removing not unused functions. 
# Clearer result printing
# Allow easier testing on own-defined datasets

import os
import sys
import time
import numpy as np
import cv2
from six.moves import xrange
from geom import *

def cal_deg_mAP(errs, thres=np.arange(5, 21, 5), to_deg=True):
    if to_deg:
        errs = np.array(errs) * 180.0 / np.pi
    pass_ratio = [np.sum(errs < th) / len(errs) for th in thres ]
    mAP = {th:np.mean(pass_ratio[:i+1]) for i, th in enumerate(thres)}
    return mAP

def eval_preds(xs, ys, Rs, ts, e_preds, y_preds, top_k=-1):
    num_sample = len(xs)
    eval_res = {}
    method_list = ['ours', 'ours_ransac', 'ransac']
    measure_list = ['err_q', 'err_t', 'inlier_ratio', 'time']
    for method in method_list:
        eval_res[method] = {}
        for measure in measure_list:
            eval_res[method][measure] = np.zeros(num_sample)

    for sid in xrange(num_sample):
        _xs = xs[sid].reshape(1, 1, -1, 4)
        _ys = ys[sid].reshape(1, -1, 2)
        _dR = Rs[sid]
        _dt = ts[sid]
        e_pred = e_preds[sid].flatten()
        y_pred = y_preds[sid].flatten()
        if len(y_pred) != _xs.shape[2]:
            y_pred = np.ones(_xs.shape[2])

        _xs = _xs.reshape(-1, 4)
        _x1 = _xs[:, :2]
        _x2 = _xs[:, 2:]
        num_pts = _xs.shape[0]

        # current validity from network
        _valid = y_pred.flatten()
        _valid_th = np.sort(_valid)[::-1][top_k]  # choose top ones (get validity threshold)

        # For every things to test
        _use_prob = False
        _probs = None
        _weighted = False

        for method in method_list:
            if method == "ours":
                _eval_func = "non-decompose"
                _mask_before = _valid >= max(0, _valid_th)
                _method = None
            elif method == "ours_ransac":
                _eval_func = "decompose"
                _mask_before = _valid >= max(0, _valid_th)
                _method = cv2.RANSAC
            elif method == 'ransac':
                _eval_func = "decompose"
                _mask_before = np.ones_like(_valid).astype(bool)
                _method = cv2.RANSAC

            # Ours
            t1 = time.time()
            if _eval_func == "non-decompose":
                _err_q, _err_t, _, _, _num_inlier, \
                    _ = eval_nondecompose(
                        _x1, _x2, e_pred, _dR, _dt, y_pred)
                _mask_after = _mask_before
            elif _eval_func == "decompose":
                _err_q, _err_t, _, _, _num_inlier, \
                    _mask_after = eval_decompose(
                        _x1, _x2, _dR, _dt, mask=_mask_before,
                        method=_method, probs=_probs,
                        weighted=_weighted, use_prob=_use_prob)
            # Load them in list
            eval_res[method]['time'][sid] = time.time() - t1
            eval_res[method]['err_q'][sid] = _err_q
            eval_res[method]['err_t'][sid] = _err_t
            eval_res[method]['inlier_ratio'][sid] = np.sum(_mask_after) / num_pts
            
    # Print results
    for method in eval_res:
        print('>>>> Method {} Sample {}'.format(method, num_sample))
        median_res = ''
        for measure in measure_list:
            median_res += '{}={:.4f} '.format(measure, np.median(eval_res[method][measure]))
        print('Median Errs: {}'.format(median_res))

        q_mAP = cal_deg_mAP(eval_res[method]['err_q'])
        t_mAP = cal_deg_mAP(eval_res[method]['err_t']) 
        eval_res[method]['q_mAP'] = q_mAP
        eval_res[method]['t_mAP'] = t_mAP
        
        mAP_res = 'Mean AP [>deg,q,t]: \n'
        for th in q_mAP:
            mAP_res += '(>{},{:.4f}, {:.4f})  '.format(th, q_mAP[th], t_mAP[th])
        print(mAP_res+'\n')
    return eval_res

def test_simple(net, data):
    import tensorflow as tf
    sess = net.sess
    x, y, R, t = net.x_in, net.y_in, net.R_in, net.t_in
    is_training = net.is_training
    logits_mean, e_hat, loss =  net.logits, net.e_hat, net.loss

    res_dir = net.res_dir_te
    config = net.config  

    print("{} Start testing on {} ".format(time.asctime(), config.data_te))

    # Data parsing
    xs = data["xs"]
    ys = data["ys"]
    Rs = data["Rs"]
    ts = data["ts"]
    
    num_sample = len(xs)
    e_preds = []
    y_preds = []  # logits
    time_us = []

    # Predict essential matrices
    for idx_cur in xrange(num_sample):
        # Use minimum kp in batch to construct the batch
        _xs = np.array(xs[idx_cur]).reshape(1, 1, -1, 4)
        _ys = np.array(ys[idx_cur]).reshape(1, -1, 2)
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
            "loss": loss
        }

        t1 = time.time()
        res = sess.run(fetch, feed_dict=feed_dict)
        time_us.append(time.time() - t1)
        e_preds.append(res["e_hat"])
        y_preds.append(res["y_hat"])

    print("Finished computing essential matrix by network, samples {}, median/mean time per sample: {:.4f}/{:.4f}s".format(
                                        num_sample, np.median(time_us), np.mean(time_us)))
    
    # Evaluate predictions
    eval_res = eval_preds(xs, ys, Rs, ts, e_preds, y_preds, config.obj_top_k)
    
    if config.sav_res_npy:
        sav_res_path = os.path.join(config.test_log_dir, '{}_test_res.npy'.format(config.data_te))
        np.save(sav_res_path, eval_res)
        print('Save final results to ', sav_res_path)
    return eval_res
