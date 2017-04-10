#!/usr/bin/env python3

import os
import sys
import time
import logging
import sqlite3
import collections
import numpy
import cvxpy
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold


DATA_FOLDER = '../data'
RESULTS_DB = './results.db'
C1_START = -20
C1_END = 25
C1_STEP = 5
C1_FORCE = None
C2_START = -20
C2_END = 25
C2_STEP = 5
C2_FORCE = None
K = 15


def create_active_vec(X):
    activity = []
    for line in X:
        y = numpy.zeros(216)
        unique = {}
        for trigram in line:
            if trigram == 0:
                continue
            unique.setdefault(trigram, 0)
            unique[trigram] += 1
        unique = sorted(unique.items(), key=lambda item: item[1], reverse=True)
        unique_len = len(unique)
        top_values = []
        index = 0
        count = 0
        while index < unique_len and count < K:
            count += unique[index][1]
            top_values.append(unique[index][0])
            index += 1
        last_trigram = unique[index - 1][0]
        last_count = unique[index - 1][1] - (count - K)
        for index, trigram in enumerate(line):
            if trigram == 0:
                continue
            if trigram == last_trigram and last_count > 0:
                last_count -= 1
                y[index] = 1
            elif trigram != last_trigram and trigram in top_values:
                y[index] = 1
        activity.append(y)
    activity = numpy.matrix(activity)
    return activity


def sum_squares(vector):
    result = 0
    for i in vector:
        result += i * i
    return result


def train_predict(X, y, z, train_index, test_index, C1, C2=float('-inf'), ws=False):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = numpy.matrix(y[train_index]), y[test_index]
    z_train, z_test = (numpy.matrix(z[train_index]).transpose(),
                       numpy.matrix(z[test_index]).transpose())

    A = create_active_vec(X_train)
    w = cvxpy.Variable(216)
    constraints = [w >= 0]

    rmseerence = 2 ** C2 * cvxpy.sum_squares(A * w - z_train)
    logregression = 2 ** C1 * cvxpy.norm(w, 1) + cvxpy.sum_entries(
        cvxpy.logistic(cvxpy.diag((X_train * w) * (-y_train)))
    )
    objective = cvxpy.Minimize(logregression + rmseerence)

    prob = cvxpy.Problem(objective, constraints)
    prob.solve(solver=cvxpy.SCS, warm_start=ws)

    predicted_test = numpy.matrix(X[test_index]) * w.value
    expected_test = y_test
    fpr_test, tpr_test, threesholds = roc_curve(expected_test, predicted_test)

    predicted_train = numpy.matrix(X[train_index]) * w.value
    expected_train = numpy.squeeze(numpy.asarray(y_train))
    fpr_train, tpr_train, threesholds = roc_curve(expected_train,
                                                  predicted_train)

    number = 0
    for i in w.value:
        if i > 1e-06:
            number += 1

    C = create_active_vec(X_test)
    size = z_test.size
    norm_rmse = w.value.sum()
    w.value = w.value / norm_rmse
    vector = numpy.array((C * w.value - z_test).transpose())[0]
    rmse = (sum_squares(vector) / size) ** 0.5

    return fpr_test, tpr_test, fpr_train, tpr_train, rmse, number


def mine_data(filename, db, experiment_id):
    base_name = os.path.basename(filename).split('.', 1)[0]
    table_name = 'results_{}_txt'.format(base_name)
    init_table(db, table_name)
    data = numpy.loadtxt(filename)
    X = data[:, 2:218]  # features

    y = data[:, 0]  # classes
    skf = StratifiedKFold(y, n_folds=10)

    z = data[:, 1]  # activities
    for ind, elem in enumerate(z):
        if elem == -1:
            z[ind] = 0

    C1_range = C1_FORCE or numpy.arange(C1_START, C1_END, C1_STEP)
    C2_range = C2_FORCE or numpy.arange(C2_START, C2_END, C2_STEP)
    if isinstance(C1_range, dict):
        C1_range = C1_range.get(filename, numpy.arange(C1_START, C1_END, C1_STEP))
    if not isinstance(C1_range, collections.Iterable):
        C1_range = [C1_range]
    if isinstance(C2_range, dict):
        C2_range = C2_range.get(filename, numpy.arange(C2_START, C2_END, C2_STEP))
    if not isinstance(C2_range, collections.Iterable):
        C2_range = [C2_range]

    for C1 in C1_range:
        for C2 in C2_range:
            roc_auc_test = []
            roc_auc_train = []
            roc_rmse = []
            number_list = []
            ind = 0
            for train_index, test_index in skf:
                ws = ind != 0
                predict_results = train_predict(X, y, z, train_index,
                                                test_index, C1, C2, ws)
                (fpr_test, tpr_test,
                 fpr_train, tpr_train,
                 rmse, number) = predict_results
                roc_auc_test.append(auc(fpr_test, tpr_test))
                roc_auc_train.append(auc(fpr_train, tpr_train))
                roc_rmse.append(rmse)
                number_list.append(number)
                ind += 1
            auc_test_mean = numpy.mean(roc_auc_test)
            auc_train_mean = numpy.mean(roc_auc_train)
            rmse_mean = numpy.mean(roc_rmse)
            number = int(round(numpy.mean(number_list)))
            query = '''INSERT INTO {} (
    experiment_id, date,
    C1, C2,
    AUC_TEST, AUC_TRAIN,
    RMSE, FEATURES)
VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?
)'''.format(table_name)
            args = (experiment_id, time.time(),
                    float(C1), float(C2),
                    auc_test_mean, auc_train_mean,
                    rmse_mean, number)
            run_query(db, query, args, commit=True)
            logging.info('For C1 = {} and C2 = {} AUC_test is {:.6g}, '
                         'AUC_train is {:.6g}, features is {:d} and '
                         'RMSE is {:.6g}'.format(C1, C2, auc_test_mean,
                                                 auc_train_mean, number,
                                                 rmse_mean))


def run_query(db, query, args=(), ret=None, commit=False):
    result = None

    cur = db.cursor()
    cur.execute(query, args)
    if ret == 'id':
        result = cur.lastrowid
    elif ret == 'all':
        result = cur.fetchall()
    elif ret == 'one':
        result = cur.fetchone()
    if commit:
        db.commit()
    cur.close()

    return result


def init_table(db, table_name):
    query = '''CREATE TABLE IF NOT EXISTS {} (
result_id INTEGER PRIMARY KEY AUTOINCREMENT,
experiment_id INTEGER,
date INTEGER,
C1 REAL,
C2 REAL,
AUC_TEST REAL,
AUC_TRAIN REAL,
RMSE REAL,
FEATURES INTEGER)'''.format(table_name)
    run_query(db, query, commit=True)


def main(argv):
    file_formatter = logging.Formatter('[{asctime}] {message}',
                                       '%d.%m.%Y %H:%M:%S', '{')
    file_handler = logging.FileHandler('mine_data.log', 'w')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('{message}', style='{')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)

    params = argv[1:]
    files = []
    for d_path in params:
        if os.path.isfile(d_path):
            files.append(d_path)
        elif os.path.isdir(d_path):
            n_files = os.listdir(d_path)
            for n_file in n_files:
                files.append(os.path.join(d_path, n_file))
        else:
            logging.critical('Invalid argument<{}>'.format(d_path))
            return 1
    if not files:
        n_files = os.listdir(DATA_FOLDER)
        for n_file in n_files:
            files.append(os.path.join(DATA_FOLDER, n_file))
    files = sorted(files, key=os.path.getsize)

    db = sqlite3.connect(RESULTS_DB)
    query = '''CREATE TABLE IF NOT EXISTS experiments (
experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
start_time INTEGER,
end_time INTEGER)'''
    run_query(db, query, commit=True)

    experiment_id = run_query(db, '''INSERT INTO experiments (
    start_time, end_time)
VALUES (?, ?)''', (time.time(), None), ret='id')

    exc = None

    try:
        for filename in files:
            logging.info('Working with <{}>:'.format(filename))
            mine_data(filename, db, experiment_id)
            logging.info('-' * 30)
    except Exception as e:
        raise Exception(e)
    finally:
        query = '''UPDATE experiments SET end_time=? WHERE experiment_id=?'''
        run_query(db, query, (time.time(), experiment_id), commit=True)
        db.close()

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print('Interrupted...')
