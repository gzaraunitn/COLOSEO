from typing import Dict, List
from einops import rearrange, repeat
import torch
import torch.distributed as dist
from random import random
from torch import nn
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch.nn.functional as F
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
from os.path import join
from sklearn.metrics import accuracy_score
from time import time


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


# class GatherLayer(torch.autograd.Function):
#     """Gathers tensors from all processes, supporting backward propagation."""
#
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         if dist.is_available() and dist.is_initialized():
#             output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
#             dist.all_gather(output, input)
#         else:
#             output = [input]
#         return tuple(output)
#
#     @staticmethod
#     def backward(ctx, *grads):
#         (input,) = ctx.saved_tensors
#         if dist.is_available() and dist.is_initialized():
#             grad_out = torch.zeros_like(input)
#             grad_out[:] = grads[dist.get_rank()]
#         else:
#             grad_out = grads[0]
#         return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


def accuracy_at_k(output, target, top_k=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k."""

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.
    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.
    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n

    if not isinstance(value, float):
        return value.squeeze(0)
    else:
        return value


def compute_entropy(x):
    """Computes entropy of input x."""
    epsilon = 1e-5
    H = -x * torch.log(x + epsilon)
    H = torch.sum(H, dim=1)
    return H


def normalize_01(vector, dim=0):
    # vector -= vector.min(dim, keepdim=True)[0]
    # vector /= vector.max(dim, keepdim=True)[0]
    # return vector
    return (vector + 1) / 2


@torch.no_grad()
def gen_positives_mask(Y1, Y2) -> torch.Tensor:
    labels_matrix = repeat(Y1, "b -> b c", c=Y2.size(0)) == Y2
    return labels_matrix


def apply_filtering(label, n_classes):
    b = label.size(0)
    res = label.clone().cuda()
    supp = torch.arange(n_classes, n_classes + b).cuda()
    mask = (res >= n_classes).cuda()
    res[mask] = supp[mask]
    return res


def is_reject(max_epochs, current_epoch):
    epochs = torch.arange(max_epochs).float()
    epochs_probs = epochs - epochs.min(0, keepdim=True)[0]
    epochs_probs /= epochs_probs.max(0, keepdim=True)[0]
    prob = epochs_probs[current_epoch]
    r = random()
    return r <= prob


def compute_ovanet_metrics(accuracy_per_class):
    closed_acc = accuracy_per_class[:-1].mean()
    open_acc = accuracy_per_class[-1]
    h_score = (2 * closed_acc * open_acc) / (closed_acc + open_acc)
    return h_score, closed_acc, open_acc


def get_curve_online(known, novel, stypes=["Bas"]):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1 :] = tp[stype][l]
                fp[stype][l + 1 :] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1 :] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1 :] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - 0.95).argmin()
        tnr_at_tpr95[stype] = 1.0 - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    correct = pred == labels
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1 :].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR


def compute_auroc(x1, x2):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, ["Bas"])
    tpr = np.concatenate([[1.0], tp["Bas"] / tp["Bas"][0], [0.0]])
    fpr = np.concatenate([[1.0], fp["Bas"] / fp["Bas"][0], [0.0]])
    res = 100.0 * (-np.trapz(1.0 - fpr, tpr))
    return res


def compute_open_set_metrics(x1, x2):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, ["Bas"])

    results = dict()

    # TNR
    mtype = "TNR"
    results[mtype] = 100.0 * tnr_at_tpr95["Bas"]

    # AUROC
    mtype = "AUROC"
    tpr = np.concatenate([[1.0], tp["Bas"] / tp["Bas"][0], [0.0]])
    fpr = np.concatenate([[1.0], fp["Bas"] / fp["Bas"][0], [0.0]])
    results[mtype] = 100.0 * (-np.trapz(1.0 - fpr, tpr))

    # DTACC
    mtype = "DTACC"
    results[mtype] = 100.0 * (
        0.5 * (tp["Bas"] / tp["Bas"][0] + 1.0 - fp["Bas"] / fp["Bas"][0]).max()
    )

    # AUIN
    mtype = "AUIN"
    denom = tp["Bas"] + fp["Bas"]
    denom[denom == 0.0] = -1.0
    pin_ind = np.concatenate([[True], denom > 0.0, [True]])
    pin = np.concatenate([[0.5], tp["Bas"] / denom, [0.0]])
    results[mtype] = 100.0 * (-np.trapz(pin[pin_ind], tpr[pin_ind]))

    # AUOUT
    mtype = "AUOUT"
    denom = tp["Bas"][0] - tp["Bas"] + fp["Bas"][0] - fp["Bas"]
    denom[denom == 0.0] = -1.0
    pout_ind = np.concatenate([[True], denom > 0.0, [True]])
    pout = np.concatenate([[0.0], (fp["Bas"][0] - fp["Bas"]) / denom, [0.5]])
    results[mtype] = 100.0 * (np.trapz(pout[pout_ind], 1.0 - fpr[pout_ind]))

    return results


def permute_clips(clip_feats):
    new_batch = []
    for i in range(clip_feats.size(0)):
        indices = torch.randperm(clip_feats.size(1))
        sample = clip_feats[i, indices]
        new_batch.append(sample)
    new_batch = torch.stack(new_batch, dim=0).cuda(device=clip_feats.device)
    return new_batch


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def compute_mahalanobis_distance(x, feats):
    x = x.cpu().numpy()
    feats = feats.detach().cpu().numpy()
    emp_cov = EmpiricalCovariance().fit(x)
    print(emp_cov)
    print(emp_cov.mahalanobis(feats))
    exit()


def sample_estimator(model, num_classes, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    model.eval()
    model.cuda()
    group_lasso = EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for j in range(num_classes):
        list_features.append(0)

    with torch.no_grad():
        for _, data, target, _, _, _ in tqdm(train_loader):
            total += data.size(0)
            data = data.cuda()
            out = model(data)
            output = out["pred"]
            out_features = out["feat"]

            # compute the accuracy
            _, pred = output.data.max(1)
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()

            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    list_features[label] = out_features[i].view(1, -1)
                else:
                    list_features[label] = torch.cat(
                        (list_features[label], out_features[i].view(1, -1)), 0
                    )
                num_sample_per_class[label] += 1

        sample_class_mean = torch.Tensor(num_classes, list_features[0].size(1)).cuda()
        for j in range(num_classes):
            sample_class_mean[j] = torch.mean(list_features[j], 0)

    x = 0
    for i in range(num_classes):
        if i == 0:
            x = list_features[i] - sample_class_mean[i]
        else:
            x = torch.cat((x, list_features[i] - sample_class_mean[i]), 0)

    # find inverse
    group_lasso.fit(x.cpu().numpy())
    precision = group_lasso.precision_
    precision = torch.from_numpy(precision).float().cuda()

    return sample_class_mean, precision


def compute_mahalanobis_score(model, test_loader, num_classes, sample_mean, precision):
    model.eval()
    model.cuda()

    scores = []
    targets = []
    predicted_classes = []
    with torch.no_grad():
        for data, target in tqdm(test_loader):

            data, target = data.cuda(), target.cuda()

            out = model(data)
            out_features = out["feat"]

            out_features = out_features.view(
                out_features.size(0), out_features.size(1), -1
            )
            out_features = torch.mean(out_features, 2)

            # compute Mahalanobis score
            gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[i]
                zero_f = out_features.data - batch_sample_mean
                term_gau = (
                    -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                )
                if i == 0:
                    gaussian_score = term_gau.view(-1, 1)
                else:
                    gaussian_score = torch.cat(
                        (gaussian_score, term_gau.view(-1, 1)), 1
                    )

            # Input_processing
            _, sample_pred = gaussian_score.max(1)
            batch_sample_mean = sample_mean.index_select(0, sample_pred)
            zero_f = out_features - batch_sample_mean
            pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
            scores.append(pure_gau)
            targets.append(target)
            predicted_classes.append(sample_pred)

        scores = torch.cat(scores)
        targets = torch.cat(targets)
        preds = torch.cat(predicted_classes)
        return scores, targets, preds


def k_means(model, train_loader, test_loader, k, return_results=False, aug_based=False):

    res = {}

    model.cuda()
    model.eval()

    feats = []
    if aug_based:
        with torch.no_grad():
            print("K-means: loading training set")
            for data, _, target in tqdm(train_loader):
                data, target = data.cuda(), target.cuda()

                out = model(data)
                feat = out["feat"]
                if model.kwargs["kmeans_project"]:
                    feat = model.kmeans_projector(feat)
                feats.append(feat.reshape(feat.size(-1)).cpu().numpy())
    else:
        with torch.no_grad():
            print("K-means: loading training set")
            for data, target in tqdm(train_loader):
                data, target = data.cuda(), target.cuda()

                out = model(data)
                feat = out["feat"]
                if model.kwargs["kmeans_project"]:
                    feat = model.kmeans_projector(feat)
                feats.append(feat.reshape(feat.size(-1)).cpu().numpy())

    print("Performing clustering...")
    kmeans = KMeans(n_clusters=k, random_state=0).fit(feats)
    print("Clustering completed")
    centroids = kmeans.cluster_centers_
    centroids = torch.tensor(centroids).float().cuda()
    centroids = F.normalize(centroids, dim=-1)

    res["centroids"] = centroids

    if return_results:

        scores = []
        targets = []
        with torch.no_grad():
            print("Testing")
            for data, target in tqdm(test_loader):
                data, target = data.cuda(), target.cuda()

                out = model(data)
                feat = out["feat"]
                if model.kwargs["kmeans_project"]:
                    feat = model.kmeans_projector(feat)
                feat = F.normalize(feat, dim=-1)
                score, _ = (feat @ centroids.T).max(dim=1)
                scores.append(score)
                targets.append(target)

        scores = torch.cat(scores)
        targets = torch.cat(targets)
        res["scores"] = scores
        res["targets"] = targets

    model.train()
    return res


def extract_features(model, loader):

    model.cuda()
    model.eval()

    feats = []
    targets = []
    with torch.no_grad():
        print("Extracting features")
        for data, target in tqdm(loader):
            data, target = data.cuda(), target.cuda()

            out = model(data)
            feat = out["feat"]
            feats.append(feat.reshape(feat.size(-1)))
            targets.append(target)
    feats = torch.stack(feats)
    targets = torch.cat(targets)
    return feats, targets


def eval_source_only(model, loader, threshold, name, n_classes=8):

    model.cuda()
    model.eval()

    correct_per_class = [0 for _ in range(n_classes + 1)]
    instances_per_class = [0 for _ in range(n_classes + 1)]

    preds = []
    gt = []

    with torch.no_grad():
        print("Evaluating")
        for data, target in tqdm(loader):
            data, target = data.cuda(), target.cuda()
            out = model(data)
            pred = out["pred"]
            entropy = compute_entropy(F.softmax(pred, dim=1))
            reject = entropy > threshold
            pred_label = pred.argmax(dim=1)
            pred_label[reject] = n_classes
            preds += pred_label.tolist()
            gt += target.tolist()
            for index in range(target.size(0)):
                label = target[index]
                predicted_label = pred_label[index]
                instances_per_class[label] += 1
                if label.item() == predicted_label.item():
                    correct_per_class[label] += 1

    val_accuracy = accuracy_score(preds, gt)
    accuracy_per_class = np.array(correct_per_class) / np.array(instances_per_class)
    closed_accuracy = accuracy_per_class[:n_classes].mean()
    open_accuracy = accuracy_per_class[-1]
    h_score = 2 * closed_accuracy * open_accuracy / (closed_accuracy + open_accuracy)

    print("VALIDATION ACCURACY: {}".format(val_accuracy))
    print("CLOSED ACCURACY: {}".format(closed_accuracy))
    print("OPEN ACCURACY: {}".format(open_accuracy))
    print("H-SCORE: {}".format(h_score))

    with open("eval_source_only.txt", "a") as logfile:
        logfile.write("{}\n".format(name))
        logfile.write("VALIDATION ACCURACY: {}\n".format(val_accuracy))
        logfile.write("CLOSED ACCURACY: {}\n".format(closed_accuracy))
        logfile.write("OPEN ACCURACY: {}\n".format(open_accuracy))
        logfile.write("H-SCORE: {}\n\n".format(h_score))


def compute_acc(y, y_hat):
    """Computes accuracy."""

    acc = y == y_hat
    if len(acc) > 0:
        acc = acc.sum().detach().true_divide(acc.size(0))
    else:
        acc = torch.tensor(0.0, device=y.device)

    return acc


def cpu_benchmark(loader):
    start = time()
    with torch.no_grad():
        print("Testing")
        for x in tqdm(loader):
            a = 1
    total = time() - start
    return total


def test(model, loader, args):

    model = model.cuda()
    model.eval()

    correct_per_class = [0 for _ in range(model.n_classes + 1)]
    instances_per_class = [0 for _ in range(model.n_classes + 1)]
    preds = []
    gt = []

    with torch.no_grad():
        print("Evaluating")
        for data, target in tqdm(loader):
            data, target = data.cuda(), target.cuda()
            out = model(data)

            feat = out["feat"]

            if args.open_set:
                pred = model.C2(feat)
            else:
                pred = out["pred"]

            if args.open_set:
                pred_label = pred.argmax(dim=1)
                preds += pred_label.tolist()
                gt += target.tolist()
                for index in range(target.size(0)):
                    label = target[index]
                    predicted_label = pred_label[index]
                    instances_per_class[label] += 1
                    if label.item() == predicted_label.item():
                        correct_per_class[label] += 1

    if args.open_set:

        val_accuracy = accuracy_score(preds, gt)
        accuracy_per_class = np.array(correct_per_class) / np.array(instances_per_class)
        closed_accuracy = accuracy_per_class[: model.n_classes].mean()
        open_accuracy = accuracy_per_class[-1]
        h_score = (
            2 * closed_accuracy * open_accuracy / (closed_accuracy + open_accuracy)
        )

        print("VALIDATION ACCURACY: {}".format(val_accuracy))
        print("CLOSED ACCURACY: {}".format(closed_accuracy))
        print("OPEN ACCURACY: {}".format(open_accuracy))
        print("H-SCORE: {}".format(h_score))


def plot_separation(model, loader, centroids, n_class, path, epoch, n_bins=50):

    model = model.cuda()
    model.eval()

    scores = []
    targets = []
    with torch.no_grad():
        print("Testing")
        for data, target in tqdm(loader):
            data, target = data.cuda(), target.cuda()

            out = model(data)
            feat = out["feat"]
            feat = F.normalize(feat, dim=-1)
            score, _ = (feat @ centroids.T).max(dim=1)
            scores.append(score)
            targets.append(target)

    scores = torch.cat(scores).tolist()
    targets = torch.cat(targets).tolist()

    model.train()

    assert len(scores) == len(targets)

    known = []
    unknown = []
    known_labels = []
    for score, label in zip(scores, targets):
        if label < n_class:
            known.append(score)
            known_labels.append(label)
        else:
            unknown.append(score)

    fig, ax = plt.subplots()
    ax.hist(
        known, n_bins, None, ec="red", fc="none", lw=1.5, histtype="step", label="known"
    )
    ax.hist(
        unknown,
        n_bins,
        None,
        ec="green",
        fc="none",
        lw=1.5,
        histtype="step",
        label="ood",
    )
    ax.legend(loc="upper left")
    save_path = join(path, "epoch_{}.jpg".format(epoch))
    plt.savefig(save_path)
    plt.close()


def compute_class_means_distance(
    model, train_loader, test_loader, return_results=False, aug_based=False
):

    res = {}

    model = model.cuda()
    model.eval()

    feats_per_class = defaultdict(list)

    if aug_based:
        with torch.no_grad():
            print("Cmeans: loading training set")
            for data, _, target in tqdm(train_loader):
                data, target = data.cuda(), target.cuda()

                out = model(data)
                feat = out["feat"]
                feat = feat.reshape(feat.size(-1))
                feats_per_class[str(target.item())].append(feat)
    else:
        with torch.no_grad():
            print("Cmeans: loading training set")
            for data, target in tqdm(train_loader):
                data, target = data.cuda(), target.cuda()

                out = model(data)
                feat = out["feat"]
                feat = feat.reshape(feat.size(-1))
                feats_per_class[str(target.item())].append(feat)

    print("Computing means")
    mean_per_class = []
    for cls in sorted(feats_per_class.keys()):
        feats = torch.stack(feats_per_class[cls])
        mean = torch.mean(feats, dim=0)
        mean_per_class.append(mean)
    mean_per_class = torch.stack(mean_per_class)
    mean_per_class = F.normalize(mean_per_class, dim=-1)

    res["means"] = mean_per_class

    if return_results:

        print("Testing")
        scores = []
        targets = []
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.cuda(), target.cuda()

                out = model(data)
                feat = out["feat"]
                feat = F.normalize(feat, dim=-1)
                score, _ = (feat @ mean_per_class.T).max(dim=1)
                scores.append(score)
                targets.append(target)

        scores = torch.cat(scores)
        targets = torch.cat(targets)
        res["scores"] = scores
        res["targets"] = targets
        res["max_similarity"] = torch.max(scores)
        res["min_similarity"] = torch.min(scores)

    return res


def compute_thresholds(model, train_loader, test_loader, aug_based=False, n_classes=6):

    model = model.cuda()
    model.eval()

    feats_per_class = defaultdict(list)

    if aug_based:
        with torch.no_grad():
            print("Thresholds: loading training set")
            for data, _, target in tqdm(train_loader):
                data, target = data.cuda(), target.cuda()

                out = model(data)
                feat = out["feat"]
                feat = feat.reshape(feat.size(-1))
                feats_per_class[str(target.item())].append(feat)
    else:
        with torch.no_grad():
            print("Thresholds: loading training set")
            for data, target in tqdm(train_loader):
                data, target = data.cuda(), target.cuda()

                out = model(data)
                feat = out["feat"]
                feat = feat.reshape(feat.size(-1))
                feats_per_class[str(target.item())].append(feat)

    print("Computing means")
    mean_per_class = []
    for cls in sorted(feats_per_class.keys()):
        feats = torch.stack(feats_per_class[cls])
        mean = torch.mean(feats, dim=0)
        mean_per_class.append(mean)
    mean_per_class = torch.stack(mean_per_class)
    mean_per_class = F.normalize(mean_per_class, dim=-1)

    print("Computing thresholds")
    test_samples_per_class = defaultdict(list)
    with torch.no_grad():
        for data, target in tqdm(test_loader):

            known_classes_ind = target != n_classes
            if torch.any(known_classes_ind):
                data = data[known_classes_ind]
                target = target[known_classes_ind]
                data, target = data.cuda(), target.cuda()

                out = model(data)
                feat = out["feat"]
                feat = F.normalize(feat, dim=-1)
                score = feat @ mean_per_class[target.item()].T
                test_samples_per_class[target.item()].append(score.item())

    hardest_per_class = []
    for label in range(n_classes):
        hardest_per_class.append(min(test_samples_per_class[label]))

    hardest_per_class = torch.tensor(hardest_per_class)

    return hardest_per_class
