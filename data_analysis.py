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
    return cursor.fetchall()


def store_db_main_info(N_gates, N_swaps, depth, prob_succs, mean_f, q_vol):

    data_frame = pd.DataFrame({"N_gates": N_gates, "N_swaps": N_swaps, "depth": depth,
                               "prob_succs": prob_succs, "mean_f": mean_f, "q_vol": q_vol})

    return data_frame


# def plot_relation(data):
#     plt.scatter(data1, data2)
#     plt.savefig("")

# def clean_data_frame(data_frame):
#     # I use Quantum Volume as the harder variable to be randomly repeated
#     data_frame.drop_duplicates(subset=['v_q'], keep='first')

N_gates = []
N_swaps = []
depth = []
prob_succs = []
mean_f = []
q_vol = []

for i in range(4):

    db_path = "~/qbench/mapping_benchmarks/simple_benchs_smart_fast{i}.db".format(
        i=i+1)

    bench_info = extract_db_main_info(db_path)
    for b_i in bench_info:
        N_gates.append(b_i[0])
        N_swaps.append(b_i[1])
        depth.append(b_i[2])
        prob_succs.append(b_i[3])
        mean_f.append(b_i[4])
        q_vol.append(b_i[5])

    # pearsonr(x, y)
