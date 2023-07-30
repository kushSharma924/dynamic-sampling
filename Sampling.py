import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from scipy.spatial.distance import euclidean
import torch.nn.functional as F
from collections import Counter
import pdb
from scipy import stats
from extract_features import CIFAR100_EXTRACT_FEATURE_CLIP_new
from copy import deepcopy
from torch.distributions import Categorical
import datasets
from torch.nn.functional import softmax
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors

def diversity_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    diversityArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        features, outputs = model(data)

        # Calculate the diversity score for each sample (e.g., distance to cluster centroids or density-based measure)
        diversity_score = calculate_diversity_score(features[0])

        diversityArr += list(diversity_score)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((diversityArr, queryIndex, labelArr)).T
    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])[::-1]]  # Sort in descending order for diversity sampling
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][:args.query_batch].astype(int)  # Select top diverse samples
    labelArr = tmp_data[2].astype(int)
    queryLabelArr = tmp_data[2][:args.query_batch]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall

def calculate_diversity_score(features_list, k=5):
    # Convert each tensor in the list to a NumPy array after detaching and moving to CPU
    features_np = []
    for tensor in features_list:
        tensor = tensor.detach().cpu()
        tensor = torch.flatten(tensor)
        features_np.append(tensor.numpy())
    # Stack the NumPy arrays along the first axis to create a 2D array
    features_np = np.stack(features_np)

    # Initialize KNN model
    knn_model = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    knn_model.fit(features_np)

    # Find the k+1 nearest neighbors (the nearest neighbor will be the point itself)
    distances, indices = knn_model.kneighbors(features_np)

    # Calculate diversity score based on the average distance of k nearest neighbors
    diversity_scores = 1.0 / (1.0 + np.mean(distances[:, 1:], axis=1))

    # Optionally, normalize diversity scores to [0, 1]
    normalized_diversity_scores = (diversity_scores - np.min(diversity_scores)) / (np.max(diversity_scores) - np.min(diversity_scores))

    return normalized_diversity_scores.tolist()


def combined_active_learning(args, unlabeledloader, len_labeled_ind_train,len_unlabeled_ind_train,labeled_ind_train,
                                                                            invalidList, model, use_gpu):
    model.eval()
    embDim = 864
    nLab = args.known_class
    embedding = np.zeros([len_unlabeled_ind_train + len_labeled_ind_train, embDim * nLab])

    queryIndex = []
    data_image = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    embed_dict = {}
    # Badge Sampling - Initial selection of informative data points
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        out, outputs = model(data)
        out = torch.flatten(out[0], start_dim=1)
        out = out.data.cpu().numpy()
        batchProbs = F.softmax(outputs, dim=1).data.cpu().numpy()
        maxInds = np.argmax(batchProbs, 1)
        v_ij, predicted = outputs.max(1)
        for j in range(len(labels)):
            tmp_mean = 0
            for c in range(nLab):
                if c == maxInds[j]:
                    embedding[index[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                    tmp_embedding = deepcopy(out[j]) * (1 - batchProbs[j][c])
                    tmp_embedding = max(tmp_embedding)
                    tmp_mean += tmp_embedding
                else:
                    embedding[index[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                    tmp_embedding = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                    tmp_embedding = max(tmp_embedding)
                    tmp_mean += tmp_embedding
            tmp_class = np.array(predicted.data.cpu())[j]
            if tmp_class not in embed_dict:
                embed_dict[tmp_class] = []
            tmp_index = index[j]
            tmp_label = np.array(labels.data.cpu())[j]
            embed_dict[tmp_class].append([tmp_mean / nLab, tmp_index, tmp_label])

        data_image += data
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))


        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            if tmp_class not in S_ij:
                S_ij[tmp_class] = []
            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])

    embedding = torch.Tensor(embedding)
    # embed, embed_index = embedding.max(1)


    # AV Sampling Temperature - Further refining of the selection
    tmp_data = []
    for tmp_class in embed_dict:
        embed_dict[tmp_class] = np.array(embed_dict[tmp_class])
        activation_value = embed_dict[tmp_class][:, 0]
        if len(activation_value) < 2:
            continue

        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(np.array(activation_value).reshape(-1, 1))
        prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
        prob = prob[:, gmm.means_.argmax()]

        if tmp_class == args.known_class:
            prob = [0] * len(prob)
            prob = np.array(prob)

        if len(tmp_data) == 0:
            tmp_data = np.hstack((prob.reshape(-1, 1), embed_dict[tmp_class]))
        else:
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), embed_dict[tmp_class]))))

    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T

    queryIndex = tmp_data[2][-args.query_batch:].astype(int)
    labelArr = tmp_data[3].astype(int)

    queryLabelArr = tmp_data[3][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + len_unlabeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall

def new_core_set(args, unlabeledloader, Len_labeled_ind_train, Len_unlabeled_ind_train, model, use_gpu):
    model.eval()
    min_distances = None
    already_selected = []
    n_obs = Len_unlabeled_ind_train
    features = []
    indices = []
    labels = []

    # Extract features
    for batch_idx, (index, (data, label)) in enumerate(unlabeledloader):
        if use_gpu:
            data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            batch_features, outputs = model(data)
            batch_features = torch.flatten(batch_features[0], start_dim=1)
            features.extend(batch_features.cpu().numpy())
            indices.extend(index)
            labels.extend(label.cpu().numpy())

    features = np.array(features)

    # Select batch
    new_batch = []
    for _ in range(args.query_batch):
        if not already_selected:
            # Initialize centers with a randomly selected datapoint
            ind = np.random.choice(np.arange(n_obs))
        else:
            ind = np.argmax(min_distances)
            assert ind not in already_selected

        # Update min_distances for all examples given new cluster center.
        dist = pairwise_distances(features, features[ind].reshape(1, -1))
        if min_distances is None:
            min_distances = dist
        else:
            min_distances = np.minimum(min_distances, dist)

        new_batch.append(ind)
        already_selected.append(ind)

    # Get labels for the selected batch
    query_labels = np.array(labels)[new_batch]

    # Calculate precision and recall
    precision = len(np.where(query_labels < args.known_class)[0]) / len(query_labels)
    recall = (len(np.where(query_labels < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(np.array(labels) < args.known_class)[0]) + Len_labeled_ind_train)

    # Separate the selected indices into two lists based on the label
    selected_indices = np.array(indices)[new_batch]
    selected_known = selected_indices[np.where(query_labels < args.known_class)[0]]
    selected_unknown = selected_indices[np.where(query_labels >= args.known_class)[0]]

    return selected_known, selected_unknown, precision, recall


def random_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (_, labels)) in enumerate(unlabeledloader):
        queryIndex += index
        labelArr += list(np.array(labels.data))

    tmp_data = np.vstack((queryIndex, labelArr)).T
    np.random.shuffle(tmp_data)
    tmp_data = tmp_data.T
    queryIndex = tmp_data[0][:args.query_batch]
    labelArr = tmp_data[1]
    queryLabelArr = tmp_data[1][:args.query_batch]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def uncertainty_sampling_hybrid(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labelArr_true):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        features, outputs = model(data)

        uncertaintyArr += list(
            np.array((-torch.softmax(outputs, 1) * torch.log(torch.softmax(outputs, 1))).sum(1).cpu().data))
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((uncertaintyArr, queryIndex, labelArr)).T
    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    labelArr_true = np.array(labelArr_true)
    queryLabelArr = tmp_data[2][-args.query_batch:]

    precision = len(np.where(queryLabelArr < args.known_class)[0]) / args.query_batch

    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr_true < args.known_class)[0]) + Len_labeled_ind_train)

    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall

def dynamic_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    # Perform Uncertainty Sampling in the initial stage (low amount of unlabeled data)
    print("x: ", Len_labeled_ind_train)
    if Len_labeled_ind_train < 80:
        print("UNCERTAINTY")
        return uncertainty_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu)
    # Switch to Diversity Sampling in later stages (high amount of unlabeled data)
    else:
            print("DIVERSITY")
    return diversity_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu)

def uncertainty_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        # if args.dataset == 'mnist':
        #     data = data.repeat(1, 3, 1, 1)
        features, outputs = model(data)

        uncertaintyArr += list(
            np.array((-torch.softmax(outputs, 1) * torch.log(torch.softmax(outputs, 1))).sum(1).cpu().data))
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((uncertaintyArr, queryIndex, labelArr)).T
    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    queryLabelArr = tmp_data[2][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall

# unlabeledloader is int 800
def AV_sampling_temperature(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, outputs = model(data)
        queryIndex += index
        # my_test_for_outputs = outputs.cpu().data.numpy()
        # print(my_test_for_outputs)
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            if tmp_class not in S_ij:
                S_ij[tmp_class] = []
            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])

    # fit a two-component GMM for each class
    tmp_data = []
    for tmp_class in S_ij:
        S_ij[tmp_class] = np.array(S_ij[tmp_class])
        activation_value = S_ij[tmp_class][:, 0]
        if len(activation_value) < 2:
            continue
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(np.array(activation_value).reshape(-1, 1))
        prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
        prob = prob[:, gmm.means_.argmax()]
        if tmp_class == args.known_class:
            prob = [0] * len(prob)
            prob = np.array(prob)

        if len(tmp_data) == 0:
            tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
        else:
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))

    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T

    queryIndex = tmp_data[2][-args.query_batch:].astype(int)
    labelArr = tmp_data[3].astype(int)

    queryLabelArr = tmp_data[3][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall
