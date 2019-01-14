import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

from scipy.stats import pearsonr
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

two_q_gates = [21,
               107,
               121,
               33,
               40,
               38,
               16,
               2,
               5,
               2,
               31,
               142,
               5,
               11,
               100,
               22,
               96,
               75,
               233,
               12,
               198,
               58,
               31,
               15,
               326,
               326,
               1773,
               1263,
               1650,
               6,
               5]


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


def fit_polynomial(x, y, degree):

    param = np.polyfit(x, y, deg=degree)

    # point = np.arange(min(x), max(x), len(x)/degree)
    point = np.array([min(x), max(x)])

    f = 0
    for p in range(len(param)):
        f = f + param[p]*point**(len(param)-p)

    return point, f


def regression(x, y):

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33)

    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" %
          mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    return X_test, y_pred


def plot_relation(y, x, save_name, ylabel, xlabel):
    # fig = plt.figure()
    plt.scatter(x, y)
    # fig.suptitle('test title', fontsize=20)

    # Fitting line (regression)
    # point, f = fit_polynomial(x, y, 1)
    # plt.plot(point, f, lw=2.5, c="k", label="fit line")
    X_test, y_pred = regression(x, y)
    plt.plot(X_test, y_pred, linewidth=3, label="fit line")

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
    f_q_corr = pearsonr(df_cl.mean_f, df_cl.q_vol)
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
    ps_q_corr = pearsonr(df_cl.prob_succs, df_cl.q_vol)
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

    meas_error = meas_error.replace(".", "_")

    general_results(df_cl, t1, meas_error)

    two_q_gates_analysis(df_cl, t1, meas_error)


data_analysis("3000", "0.005")
data_analysis("1000", "0.005")
data_analysis("3000", "0")
