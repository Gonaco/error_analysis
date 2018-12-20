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

def clean_data_frame(data_frame):
    # I use Quantum Volume as the harder variable to be randomly repeated
    data_frame.drop_duplicates(subset=['q_vol'], keep='first')
    return data_frame


N_gates = []
N_swaps = []
depth = []
prob_succs = []
mean_f = []
q_vol = []

for i in range(5):

    db_path = "/home/dmorenomanzano/qbench/mapping_benchmarks/simple_benchs_smart_fast{i}.db".format(
        i=i if i > 0 else "")

    bench_info = extract_db_main_info(db_path)
    for b_i in bench_info:
        N_gates.append(b_i[0])
        N_swaps.append(b_i[1])
        depth.append(b_i[2])
        prob_succs.append(b_i[3])
        mean_f.append(b_i[4])
        q_vol.append(b_i[5])

data_frame = store_db_main_info(
    N_gates, N_swaps, depth, prob_succs, mean_f, q_vol)
df_cl = clean_data_frame(data_frame)

print("\n\t-- Correlation between Fidelity and:")

print("\n- # of Gates:")
f_g_corr = pearsonr(df_cl.mean_f, df_cl.N_gates)
print(f_g_corr)

print("\n- # of Swaps:")
f_s_corr = pearsonr(df_cl.mean_f, df_cl.N_swaps)
print(f_s_corr)

print("\n- Depth:")
f_d_corr = pearsonr(df_cl.mean_f, df_cl.depth)
print(f_d_corr)

print("\n- Quantum Volume:")
f_q_corr = pearsonr(df_cl.mean_f, df_cl.N_gates)
print(f_q_corr)

print("\n\n\t-- Correlation between Probability of Success and:")

print("\n- # of Gates:")
ps_g_corr = pearsonr(df_cl.prob_succs, df_cl.N_gates)
print(ps_g_corr)

print("\n- # of Swaps:")
ps_s_corr = pearsonr(df_cl.prob_succs, df_cl.N_swaps)
print(ps_s_corr)

print("\n- Depth:")
ps_d_corr = pearsonr(df_cl.prob_succs, df_cl.depth)
print(ps_d_corr)

print("\n- Quantum Volume:")
ps_q_corr = pearsonr(df_cl.prob_succs, df_cl.N_gates)
print(ps_q_corr)
