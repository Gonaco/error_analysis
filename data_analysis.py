import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

from scipy.stats import pearsonr
import pandas as pd


def extract_db_main_info(db_path):

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    query = "SELECT DISTINCT HardwareBenchs.N_gates, HardwareBenchs.N_swaps, depth, prob_succs, mean_f, q_vol FROM SimulationsInfo LEFT JOIN HardwareBenchs ON algorithm=HardwareBenchs.id LEFT JOIN Results ON result=Results.id LEFT JOIN Experiments ON experiment=Experiments.id LEFT JOIN Benchmarks ON HardwareBenchs.benchmark=Benchmarks.id LEFT JOIN Configurations ON configuration=Configurations.id;"
    cursor.execute(query)
    return cursor.fetchone()

# def store_db_main_info(main_info):

# def plot_relation(data):
#     plt.scatter(data1, data2)
#     plt.savefig("")


# def clean_data_frame(data_frame):
#     # I use Quantum Volume as the harder variable to be randomly repeated
#     data_frame.drop_duplicates(subset=['v_q'], keep='first')


for i in range(4):

    db_path = "~/qbench/mapping_benchmarks/simple_benchs_smart_fast/simple_benchs_smart_fast{i}.db".format(
        i=i+1)

    bench_info = extract_db_main_info(db_path)
    print(bench_info)

    # pearsonr(x, y)
