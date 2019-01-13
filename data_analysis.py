import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

from scipy.stats import pearsonr
import pandas as pd


def extract_decoher_info(db_path, t1):

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    query = "SELECT DISTINCT HardwareBenchs.N_gates, HardwareBenchs.N_swaps, depth, prob_succs, mean_f, q_vol FROM SimulationsInfo LEFT JOIN HardwareBenchs ON algorithm=HardwareBenchs.id LEFT JOIN Results ON result=Results.id LEFT JOIN Experiments ON experiment=Experiments.id LEFT JOIN Benchmarks ON HardwareBenchs.benchmark=Benchmarks.id LEFT JOIN Configurations ON configuration=Configurations.id WHERE SimulationsInfo.t1 = {t1} AND initial_placement='no' AND Benchmarks.N_gates < 3888;"
    cursor.execute(query.format(t1=t1))
    return cursor.fetchall()


def extract_info(db_path, t1, meas_error):

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # NO INITIAL PLACEMENT IS TO AVOID FOR NOW THE ERROR OF INITIAL PLACEMENTS AND THE NUMBER OF GATES TO AVOID THE ALGORITHM SYM6
    query = "SELECT DISTINCT HardwareBenchs.N_gates, HardwareBenchs.N_swaps, depth, prob_succs, mean_f, q_vol FROM SimulationsInfo LEFT JOIN HardwareBenchs ON algorithm=HardwareBenchs.id LEFT JOIN Results ON result=Results.id LEFT JOIN Experiments ON experiment=Experiments.id LEFT JOIN Benchmarks ON HardwareBenchs.benchmark=Benchmarks.id LEFT JOIN Configurations ON configuration=Configurations.id WHERE SimulationsInfo.t1 = {t1} AND meas_error = {meas_error} AND initial_placement='no' AND Benchmarks.N_gates < 3888;"
    cursor.execute(query.format(t1=t1, meas_error=meas_error))
    return cursor.fetchall()


def store_db_main_info(N_gates, N_swaps, depth, prob_succs, mean_f, q_vol):

    data_frame = pd.DataFrame({"N_gates": N_gates, "N_swaps": N_swaps, "depth": depth,
                               "prob_succs": prob_succs, "mean_f": mean_f, "q_vol": q_vol})

    return data_frame


def plot_relation(y, x, save_name, ylabel, xlabel, degree=1):
    # fig = plt.figure()
    plt.scatter(x, y)
    # fig.suptitle('test title', fontsize=20)

    # Fitting line (regression)
    a, b = np.polyfit(x, y, deg=degree)

    point = np.array([min(x), max(x)])
    plt.plot(point, a*point + b, lw=2.5, c="k", label="fit line")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_name)
    plt.clf()


def clean_data_frame(data_frame):
    # I use Quantum Volume as the harder variable to be randomly repeated
    data_frame.drop_duplicates(subset=['q_vol'], keep='first')
    return data_frame


def general_results(df_cl, t1, meas_error):

    print("\n\t-- Correlation between Fidelity and:")

    print("\n- # of Gates:")
    f_g_corr = pearsonr(df_cl.mean_f, df_cl.N_gates)
    plot_relation(df_cl.mean_f, df_cl.N_gates,
                  "f_g_"+t1+"_"+meas_error, "fidelity", "# of gates")
    print(f_g_corr)

    print("\n- # of Swaps:")
    f_s_corr = pearsonr(df_cl.mean_f, df_cl.N_swaps)
    plot_relation(df_cl.mean_f, df_cl.N_swaps,
                  "f_s_"+t1+"_"+meas_error, "fidelity", "# of swaps")
    print(f_s_corr)

    print("\n- Depth:")
    f_d_corr = pearsonr(df_cl.mean_f, df_cl.depth)
    plot_relation(df_cl.mean_f, df_cl.depth, "f_d_" +
                  t1+"_"+meas_error, "fidelity", "depth")
    print(f_d_corr)

    print("\n- Quantum Volume:")
    f_q_corr = pearsonr(df_cl.mean_f, df_cl.N_gates)
    plot_relation(df_cl.mean_f, df_cl.q_vol, "f_q_" +
                  t1+"_"+meas_error, "fidelity", "V_Q")
    print(f_q_corr)

    print("\n\n\t-- Correlation between Probability of Success and:")

    print("\n- # of Gates:")
    ps_g_corr = pearsonr(df_cl.prob_succs, df_cl.N_gates)
    plot_relation(df_cl.prob_succs, df_cl.N_gates,
                  "ps_g_"+t1+"_"+meas_error, "prob. success", "# of gates")
    print(ps_g_corr)

    print("\n- # of Swaps:")
    ps_s_corr = pearsonr(df_cl.prob_succs, df_cl.N_swaps)
    plot_relation(df_cl.prob_succs, df_cl.N_swaps,
                  "ps_s_"+t1+"_"+meas_error, "prob. success", "# of swaps")
    print(ps_s_corr)

    print("\n- Depth:")
    ps_d_corr = pearsonr(df_cl.prob_succs, df_cl.depth)
    plot_relation(df_cl.prob_succs, df_cl.depth,
                  "ps_d_"+t1+"_"+meas_error, "prob. success", "depth")
    print(ps_d_corr)

    print("\n- Quantum Volume:")
    ps_q_corr = pearsonr(df_cl.prob_succs, df_cl.N_gates)
    plot_relation(df_cl.prob_succs, df_cl.q_vol,
                  "ps_q_"+t1+"_"+meas_error, "prob. success", "V_Q")
    print(ps_q_corr)


def fidelity_diff(df_cl):

    f_diff_array = []
    N_swaps = []

    for index, row in df_cl.iterrows():

        if row["N_swaps"] == 0:
            no_map_entr = row["mean_f"]
        else:
            f_diff_array.append(no_map_entr - row["mean_f"])
            N_swaps.append(row["N_swaps"])

    return f_diff_array, N_swaps


def two_q_gates_analysis(df_cl, t1, meas_error):

    f_diff_array, N_swaps = fidelity_diff(df_cl)

    print("\n\t-- Correlation between the decrement in Fidelity and # of SWAPS")

    f_s_corr = pearsonr(f_diff_array, N_swaps)
    plot_relation(f_diff_array, N_swaps,
                  "f_s_2qg_"+t1+"_"+meas_error, "decrement in fidelity", "# of SWAPS")
    print(f_s_corr)


def data_analysis(t1, meas_error):

    t1 = str(t1)
    meas_error = str(meas_error)
    N_gates = []
    N_swaps = []
    depth = []
    prob_succs = []
    mean_f = []
    q_vol = []

    print("\n\tAnalysis For Decoherence Time = " +
          t1+" and Error Measurement = "+meas_error)
    print("\n\t-------------------------------")

    for i in range(5):

        db_path = "/home/dmorenomanzano/qbench/mapping_benchmarks/simple_benchs_smart_fast{i}.db".format(
            i=i if i > 0 else "")

        # bench_info = extract_decoher_info(db_path, t1)
        bench_info = extract_info(db_path, t1, meas_error)
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

    general_results(df_cl, t1, meas_error)

    two_q_gates_analysis(df_cl, t1, meas_error)


data_analysis("3000", "0.005")
data_analysis("1000", "0.005")
data_analysis("3000", "0")
