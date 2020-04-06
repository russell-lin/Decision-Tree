from typing import Optional, Any

import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    con_en = []
    frac = []
    for i in range(len(branches)):
        frac.append(sum(branches[i]))
        entropy = 0
        if frac[i] == 0:
            con_en.append(0)
        else:
            final_entropy = []
            for j in range(len(branches[i])):
                if branches[i][j] > 0:
                    entropy += (-1) * branches[i][j] / sum(branches[i]) * np.log2(branches[i][j] / sum(branches[i]))
            con_en.append(entropy)
    total = sum(frac)
    final_con_en = 0
    for i in range(len(frac)):
        final_con_en += frac[i] / total * con_en[i]
    IG = S - final_con_en
    return IG

# TODO: implement reduced error prunning function, pruning your tree on this function
def Choose_Node(decisionTree, Node, X_test, y_test):
    if Node.splittable == True:
        for child in Node.children:
            if child.splittable == True:
                Choose_Node(decisionTree, child, X_test, y_test)

        old_accuracy = 0
        predict_label = decisionTree.predict(X_test)
        for i in range(len(y_test)):
            if y_test[i] == predict_label[i]:
                old_accuracy += 1

        Node.splittable = False
        new_accuracy = 0
        predict_label_after = decisionTree.predict(X_test)
        for j in range(len(y_test)):
            if y_test[j] == predict_label_after[j]:
                new_accuracy += 1

        if new_accuracy < old_accuracy:
            Node.splittable = True
    return


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List

    Choose_Node(decisionTree, decisionTree.root_node, X_test, y_test)

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
