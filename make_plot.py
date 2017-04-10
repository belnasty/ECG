#!/usr/bin/env python3
import os
import sys
import time
import logging
import sqlite3
import numpy
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
 
 
RESULTS_DB = './results.db'
PLOTS_PATH = './plots'
POINTS_NUM = 150
AUC_MIN = 0
 
 
def make_auc_c1_graph(db, table_name, experiments):
    data = run_query(db, '''SELECT DISTINCT
   C1, AUC_TEST, AUC_TRAIN
FROM {}
WHERE experiment_id IN ({}) AND C2 = ? ORDER BY C1 ASC'''.format(table_name, experiments), (float('-inf'),), ret='all')
 
    C1_list = []
    min_C1 = float('+inf')
    max_C1 = float('-inf')
    auc_test_list = []
    auc_train_list = []
    for point in data:
        C1 = point[0]
        C1_list.append(C1)
        if C1 < min_C1:
            min_C1 = C1
        if C1 > max_C1:
            max_C1 = C1
        auc_test_list.append(point[1])
        auc_train_list.append(point[2])
 
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    params = {
        'text.usetex': True,
        'font.size': 15,
        'text.latex.unicode': True
    }
    plt.rcParams.update(params)
    linear_space = numpy.linspace(min_C1, max_C1, POINTS_NUM)
    auc_test_smooth = interp1d(C1_list, auc_test_list, kind='cubic')
    auc_train_smooth = interp1d(C1_list, auc_train_list, kind='cubic')
    plt.plot(linear_space, auc_test_smooth(linear_space), 'r')
    plt.plot(C1_list, auc_test_list, 'ro')
    plt.plot(linear_space, auc_train_smooth(linear_space), 'b')
    plt.plot(C1_list, auc_train_list, 'bo')
    plt.xlim(min_C1 - 1, max_C1 + 1)
    plt.xlabel('$\log_2 C_1$')
    plt.ylabel('AUC')
    plt.grid(True)
    path = os.path.join(PLOTS_PATH, '{}_AUC_C1.eps'.format(table_name))
    plt.savefig(path)
    plt.close()
    logging.info('Plot saved to <{}>'.format(path))
 
 
 
def make_auc_c2_graph(db, table_name, experiments):
    C1_list_raw = run_query(db, '''SELECT DISTINCT
   C1 FROM {}
WHERE experiment_id IN ({}) ORDER BY C1 ASC'''.format(table_name, experiments), ret='all')
    C1_cnt = len(C1_list_raw)
    C1_str = ''
    C1_list = []
    for ind, C1_raw in enumerate(C1_list_raw):
        C1 = C1_raw[0]
        C1_list.append(C1)
        C1_str += '{:g}'.format(C1)
        if ind + 1 < C1_cnt:
            C1_str += ', '
    logging.info('List of available C1: {}'.format(C1_str))
 
    C1 = input('Enter C1> ')
    try:
        C1 = float(C1)
    except:
        logging.critical('Incorrect C1 <{}>'.format(C1))
        return
    if not C1 in C1_list:
        logging.critical('Incorrect C1 <{}>'.format(C1))
        return
 
    data = run_query(db, '''SELECT DISTINCT
   C2, AUC_TEST, AUC_TRAIN
FROM {}
WHERE experiment_id IN ({}) AND C1 = ? AND C2 <> ? ORDER BY C2 ASC'''.format(table_name, experiments), (C1, float('-inf')), ret='all')
 
    C2_list = []
    min_C2 = float('+inf')
    max_C2 = float('-inf')
    auc_test_list = []
    auc_train_list = []
    for point in data:
        C2 = point[0]
        C2_list.append(C2)
        if C2 < min_C2:
            min_C2 = C2
        if C2 > max_C2:
            max_C2 = C2
        auc_test_list.append(point[1])
        auc_train_list.append(point[2])
 
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    params = {
        'text.usetex': True,
        'font.size': 15,
        'text.latex.unicode': True
    }
    plt.rcParams.update(params)
    linear_space = numpy.linspace(min_C2, max_C2, POINTS_NUM)
    auc_test_smooth = interp1d(C2_list, auc_test_list, kind='cubic')
    auc_train_smooth = interp1d(C2_list, auc_train_list, kind='cubic')
    plt.plot(linear_space, auc_test_smooth(linear_space), 'r')
    plt.plot(C2_list, auc_test_list, 'ro')
    plt.plot(linear_space, auc_train_smooth(linear_space), 'b')
    plt.plot(C2_list, auc_train_list, 'bo')
    plt.xlim(min_C2 - 1, max_C2 + 1)
    plt.xlabel('$\log_2 C_2$')
    plt.ylabel('AUC')
    plt.grid(True)
    path = os.path.join(PLOTS_PATH, '{}_AUC_C2.eps'.format(table_name))
    plt.savefig(path)
    plt.close()
    logging.info('Plot saved to <{}>'.format(path))
 
 
def make_rmse_graph(db, table_name, experiments):
    C1_list_raw = run_query(db, '''SELECT DISTINCT
   C1 FROM {}
WHERE experiment_id IN ({}) ORDER BY C1 ASC'''.format(table_name, experiments), ret='all')
    C1_cnt = len(C1_list_raw)
    C1_str = ''
    C1_list = []
    for ind, C1_raw in enumerate(C1_list_raw):
        C1 = C1_raw[0]
        C1_list.append(C1)
        C1_str += '{:g}'.format(C1)
        if ind + 1 < C1_cnt:
            C1_str += ', '
    logging.info('List of available C1: {}'.format(C1_str))
 
    C1 = input('Enter C1> ')
    try:
        C1 = float(C1)
    except:
        logging.critical('Incorrect C1 <{}>'.format(C1))
        return
    if not C1 in C1_list:
        logging.critical('Incorrect C1 <{}>'.format(C1))
        return
 
    data = run_query(db, '''SELECT DISTINCT
   C2, RMSE
FROM {}
WHERE experiment_id IN ({}) AND C1 = ? ORDER BY C2 ASC'''.format(table_name, experiments), (C1,), ret='all')
    C2_list = []
    min_C2 = float('+inf')
    max_C2 = float('-inf')
    rmse_list = []
    for point in data:
        C2 = point[0]
        C2_list.append(C2)
        if C2 < min_C2:
            min_C2 = C2
        if C2 > max_C2:
            max_C2 = C2
        rmse_list.append(point[1])
 
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    params = {
        'text.usetex': True,
        'font.size': 15,
        'text.latex.unicode': True
    }
    plt.rcParams.update(params)
    linear_space = numpy.linspace(min_C2, max_C2, POINTS_NUM)
    rmse_smooth = interp1d(C2_list, rmse_list, kind='cubic')
    plt.plot(linear_space, rmse_smooth(linear_space), 'b')
    plt.plot(C2_list, rmse_list, 'bo')
    plt.xlim(min_C2 - 1, max_C2 + 1)
    plt.xlabel('$\log_2 C_2$')
    plt.ylabel('RMSE')
    plt.grid(True)
    path = os.path.join(PLOTS_PATH, '{}_RMSE_C2.eps'.format(table_name))
    plt.savefig(path)
    plt.close()
    logging.info('Plot saved to <{}>'.format(path))
 
 
def make_full_auc_graph(db, table_name, experiments):
    data = run_query(db, '''SELECT DISTINCT
   C1, C2, AUC_TEST
FROM {}
WHERE experiment_id IN ({}) ORDER BY C2 ASC, C1 ASC'''.format(table_name, experiments), ret='all')
    C1_list = []
    min_C1 = float('+inf')
    max_C1 = float('-inf')
    C2_list = []
    min_C2 = float('+inf')
    max_C2 = float('-inf')
    auc_list = []
    cur_C1_list = []
    cur_C2_list = []
    cur_auc_list = []
    cur_C2 = None
    for point in data:
        print(point)
        C1 = point[0]
        C2 = point[1]
        if C2 != cur_C2 and cur_C2 != None:
            C1_list.append(cur_C1_list)
            C2_list.append(cur_C2_list)
            cur_C1_list.append(2 * cur_C1_list[-1] - cur_C1_list[-2])
            cur_C2_list.append(cur_C2_list[0])
            auc_list.append(cur_auc_list)
            cur_C1_list = []
            cur_C2_list = []
            cur_auc_list = []
        cur_C2 = C2
        cur_C1_list.append(C1)
        if C1 < min_C1:
            min_C1 = C1
        if C1 > max_C1:
            max_C1 = C1
        cur_C2_list.append(C2)
        if C2 < min_C2:
            min_C2 = C2
        if C2 > max_C2:
            max_C2 = C2
        cur_auc_list.append(point[2])
    cur_C1_list.append(2 * cur_C1_list[-1] - cur_C1_list[-2])
    cur_C2_list.append(cur_C2_list[0])
    C1_list.append(cur_C1_list)
    C1_list.append(cur_C1_list)
    C2_list.append(cur_C2_list)
    C2_list.append([2 * C2_list[-1][0] - C2_list[-2][0]] * len(cur_C2_list))
    auc_list.append(cur_auc_list)
 
    C1_list = numpy.array(C1_list)
    C2_list = numpy.array(C2_list)
    auc_list = numpy.array(auc_list)

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    params = {
        'text.usetex': True,
        'font.size': 15,
        'text.latex.unicode': True
    }
    plt.rcParams.update(params)

    plt.pcolor(C1_list, C2_list, auc_list)
    plt.colorbar()
    plt.xlabel('$\log_2 C_1$')
    plt.ylabel('$\log_2 C_2$')
    plt.grid(True)
    path = os.path.join(PLOTS_PATH, '{}_AUC_C1_C2.eps'.format(table_name))
    plt.savefig(path)
    plt.close()
    logging.info('Plot saved to <{}>'.format(path))
 
 
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
 
 
def get_tablename(db):
    tables_raw = run_query(db, '''SELECT name
FROM sqlite_master
WHERE type='table';''', ret='all')
    tables_raw = sorted(tables_raw, key=lambda t: t[0])
    tables = []
    max_ind = 1
    for table in tables_raw:
        tname = table[0]
        if tname.startswith('results_'):
            tables.append((max_ind, tname))
            max_ind += 1
    max_ind -= 1
   
    logging.info('List of available tables:')
    for ind, table in tables:
        logging.info('{}) {}'.format(ind, table))
 
    tables = dict(tables)
    ind = input('Please input table number [1-{}]> '.format(max_ind))
    try:
        ind = int(ind)
    except:
        logging.critical('Incorrect table number <{}>'.format(ind))
        return
    if not ind in tables:
        logging.critical('Incorrect table number <{}>'.format(ind))
        return
    return tables[ind]
 
 
def get_experiments_list(db, table_name):
    experiments = run_query(db, '''SELECT DISTINCT
   {0}.experiment_id,
   experiments.start_time,
   experiments.end_time
FROM {0}
INNER JOIN experiments
   ON {0}.experiment_id = experiments.experiment_id'''.format(table_name),
    ret='all')
    experiments = sorted(experiments, key=lambda ex: ex[0])
    exp_numbers = []
    logging.info('List of available experiments:')
    for experiment in experiments:
        pr_time = lambda t: time.strftime('%d.%m.%Y %H:%M:%S', time.localtime(t))
        if experiment[2]:
            logging.info('{}) Start time: {}, End time: {}'.format(experiment[0], pr_time(experiment[1]), pr_time(experiment[2])))
        else:
            logging.info('{}) Start time: {}'.format(experiment[0], pr_time(experiment[1])))
        exp_numbers.append(experiment[0])
 
    exps = input('Please enter number of experiments to use (comma separated)> ')
    for exp in exps.split(','):
        try:
            exp = int(exp)
        except:
            logging.critical('Incorrect experiment number <{}>'.format(exp))
            return
        if not exp in exp_numbers:
            logging.critical('Incorrect experiment number <{}>'.format(exp))
            return
    return exps
 
 
def main(argv):
    file_formatter = logging.Formatter('[{asctime}] {message}',
                                       '%d.%m.%Y %H:%M:%S', '{')
    file_handler = logging.FileHandler('make_plot.log', 'w')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('{message}', style='{')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)
 
    db = sqlite3.connect(RESULTS_DB)
    table_name = get_tablename(db)
    if table_name:
        logging.info('Working with <{}>'.format(table_name))
        experiments = get_experiments_list(db, table_name)
        if experiments:
            logging.info('Using <{}> experiments'.format(experiments))
            logging.info('Available plots:')
            plots = [
                ('AUC by C1 (C2=-inf)', make_auc_c1_graph),
                ('AUC by C2 (fixed C1)', make_auc_c2_graph),
                ('RMSE by C2 (fixed C1)', make_rmse_graph),
                ('AUC by C1, C2', make_full_auc_graph)
            ]
            for ind, plot in enumerate(plots):
                logging.info('{}) {}'.format(ind + 1, plot[0]))
            plot_num = input('What plot do you want? [1-{}]> '.format(len(plots)))
 
            try:
                plot_num = int(plot_num)
            except:
                logging.critical('Incorrect plot number <{}>'.format(plot_num))
            else:
                if plot_num < 1 or plot_num > len(plots):
                    logging.critical('Incorrect plot number <{}>'.format(plot_num))
                else:
                    os.makedirs(PLOTS_PATH, exist_ok=True)
                    plots[plot_num - 1][1](db, table_name, experiments)
    db.close()
    return 0
 
 
if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print('Interrupted...')