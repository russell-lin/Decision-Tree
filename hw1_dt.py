import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        self.cls_max = None
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splittable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    # TODO: try to split current node
    def split(self):
        # get each feature list
        if len(self.features) == 0:
            self.splittable = False
            return
        features = np.array(self.features)
        labels = np.array(self.labels)
        max_IG = 0
        unique_feature_split = np.array([])
        for k in range(len(self.features[0])):
            uniq_features_dict = {}
            uniq_label_dict = {}
            unique_features = np.unique(features.T[k])
            unique_labels = np.unique(labels)

            features_label = []
            counter_dict = {}
            for i in self.labels:
                if i not in counter_dict.keys():
                    counter_dict[i] = 1
                else:
                    counter_dict[i] += 1
            # calculate parents entropy
            S = 0
            for i in unique_labels:
                S += (-1) * (counter_dict[i] / len(self.labels) * np.log2(counter_dict[i] / len(self.labels)))

            for i, j in zip(self.features, self.labels):
                features_label.append((i[k], j))

            c_dict = {}
            for i in features_label:
                if i not in c_dict.keys():
                    c_dict[i] = 1
                else:
                    c_dict[i] += 1

            for i in range(len(unique_features)):
                uniq_features_dict = {unique_features[i]: i for i in range(len(unique_features))}

            for j in range(len(unique_labels)):
                uniq_label_dict = {unique_labels[i]: i for i in range(len(unique_labels))}

            branches = [[0] * len(unique_labels) for i in range(len(unique_features))]
            for x in features_label:
                branches[uniq_features_dict[x[0]]][uniq_label_dict[x[1]]] = c_dict[x]
            # calculate the information gain
            IG = Util.Information_Gain(S, branches)

            if IG > max_IG or (IG == max_IG and len(unique_features) > len(unique_feature_split)):
                max_IG = IG
                unique_feature_split = unique_features
                selected_index = k

        self.dim_split = selected_index
        self.feature_uniq_split = unique_feature_split.tolist()

        if len(self.feature_uniq_split) == 0 or max_IG == 0:
            self.splittable = False

        # split the node
        to_split = self.feature_uniq_split
        cut = self.dim_split
        for i in range(len(to_split)):
            children_features = features[features[:, cut] == to_split[i]]
            New_features = np.delete(children_features, cut, axis=1).tolist()
            New_labels = labels[features[:, cut] == to_split[i]].tolist()
            New_num_cls = len(New_labels)
            chil = TreeNode(New_features, New_labels, New_num_cls)
            self.children.append(chil)

        for child in self.children:
            if child.splittable:
                child.split()

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int

        if self.splittable and len(feature) != 0:
            index_chil = self.feature_uniq_split.index(feature[self.dim_split])
            feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
            return self.children[index_chil].predict(feature)
        else:
            return self.cls_max

