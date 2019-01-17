import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

from scipy.stats import pearsonr
import pandas as pd

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

two_q_gates = {
    "4gt11_82": 18,
    "4gt12_v1_89": 100,
    "4gt4_v0_72": 113,
    "4mod5_bdd_287": 31,
    "4mod5_v0_20": 10,
    "alu_bdd_288": 38,
    "alu_v0_27": 17,
    "benstein_vazirani_15b_secret_128": 1,
    "cnt3_5_179": 85,
    "cuccaroAdder_1b": 5,
    "cuccaroMultiplier_1b": 2,
    "decod24_bdd_294": 32,
    "decod24_enable_126": 149,
    "graycode6_47": 5,
    "ham3_102": 11,
    "hwb4_49": 107,
    "ising_model_10": 90,
    "miller_11": 23,
    "mini_alu_167": 126,
    "mini_alu_305": 77,
    "mod10_176": 78,
    "mod5adder_127": 239,
    "mod5d1_63": 13,
    "mod8_10_177": 196,
    "one_two_three_v1_99": 59,
    "one_two_three_v3_101": 32,
    "qft_10": 90,
    "rd32_v0_66": 16,
    "sf_274": 336,
    "sf_276": 336,
    "shor_15": 1788,
    "sqrt8_260": 1314,
    "squar5_261": 869,
    "square_root_7": 3089,
    "sym6_145": 1701,
    "sym6_316": 123,
    "vbeAdder_2b": 6,
    "xor5_254": 5
}


benchmark_selection_corr_ps_f = ["graycode6_47",
                                 "sf_274",
                                 "mod5d1_63"
                                 "xor5_254",
                                 "ham3_102"]


def extract_decoher_info(db_path, t1):

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    query = "SELECT DISTINCT HardwareBenchs.N_gates, HardwareBenchs.N_swaps, depth, prob_succs, mean_f, q_vol, Benchmarks.benchmark, mapper FROM SimulationsInfo LEFT JOIN HardwareBenchs ON algorithm=HardwareBenchs.id LEFT JOIN Results ON result=Results.id LEFT JOIN Experiments ON experiment=Experiments.id LEFT JOIN Benchmarks ON HardwareBenchs.benchmark=Benchmarks.id LEFT JOIN Configurations ON configuration=Configurations.id WHERE SimulationsInfo.t1 = {t1} AND initial_placement='no' AND Benchmarks.N_gates < 3888;"
    cursor.execute(query.format(t1=t1))
    return cursor.fetchall()


def extract_info(db_path, t1, meas_error):

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # NO INITIAL PLACEMENT IS TO AVOID FOR NOW THE ERROR OF INITIAL PLACEMENTS AND THE NUMBER OF GATES TO AVOID THE ALGORITHM SYM6
    query = "SELECT DISTINCT HardwareBenchs.N_gates, HardwareBenchs.N_swaps, depth, prob_succs, mean_f, q_vol, Benchmarks.benchmark, mapper FROM SimulationsInfo LEFT JOIN HardwareBenchs ON algorithm=HardwareBenchs.id LEFT JOIN Results ON result=Results.id LEFT JOIN Experiments ON experiment=Experiments.id LEFT JOIN Benchmarks ON HardwareBenchs.benchmark=Benchmarks.id LEFT JOIN Configurations ON configuration=Configurations.id WHERE SimulationsInfo.t1 = {t1} AND meas_error = {meas_error} AND initial_placement='no' AND Benchmarks.N_gates < 3888;"
    cursor.execute(query.format(t1=t1, meas_error=meas_error))
    return cursor.fetchall()


def store_db_main_info(N_gates, N_two_qg, N_swaps, depth, prob_succs, mean_f, q_vol, mapper, benchmark):

    data_frame = pd.DataFrame({"N_gates": N_gates, "N_two_qg": N_two_qg, "N_swaps": N_swaps, "depth": depth,
                               "prob_succs": prob_succs, "mean_f": mean_f, "q_vol": q_vol, "mapper": mapper, "benchmark": benchmark})

    return data_frame


def fit_polynomial(x, y, degree):

    param = np.polyfit(x, y, deg=degree)

    # point = np.arange(min(x), max(x), len(x)/degree)
    point = np.array([min(x), max(x)])

    f = 0
    for p in range(len(param)):
        f = f + param[p]*point**(len(param)-p)

    return point, f


def linear_regression(x, y):

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(x), np.array(y), test_size=0.33)

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

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


def svm_regression(x, y):

    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(x, y).predict(x)
    y_lin = svr_lin.fit(x, y).predict(x)
    y_poly = svr_poly.fit(x, y).predict(x)

    return y_rbf, y_lin, y_poly


def plot_relation(y, x, save_name, ylabel, xlabel):
    # fig = plt.figure()
    plt.scatter(x, y)
    # fig.suptitle('test title', fontsize=20)

    # Fitting line (regression)
    # point, f = fit_polynomial(x, y, 1)
    # plt.plot(point, f, lw=2.5, c="k", label="fit line")

    X_test, y_pred = linear_regression(x, y)
    plt.plot(X_test, y_pred, linewidth=3, label="fit line (linear regression)")

    y_rbf, y_lin, y_poly = svm_regression(x, y)
    plt.plot(x, y_rbf, color='navy', lw=3, label='RBF model')
    plt.plot(x, y_lin, color='c', lw=3, label='Linear model')
    plt.plot(x, y_poly, color='orange', lw=3, label='Polynomial model')

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


def prb_succs_diff(df_cl):

    ps_diff_array = []
    N_swaps = []

    for index, row in df_cl.iterrows():

        if row["N_swaps"] == 0:
            no_map_entr = row["prob_succs"]
        else:
            ps_diff_array.append(no_map_entr - row["prob_succs"])
            N_swaps.append(row["N_swaps"])

    return ps_diff_array, N_swaps


def two_q_gates_f_diff_analysis(df_cl, t1, meas_error):

    f_diff_array, N_swaps = fidelity_diff(df_cl)
    ps_diff_array, N_swaps = prb_succs_diff(df_cl)

    print("\n\t-- Correlation between the decrement in Fidelity and # of SWAPS")

    f_s_corr = pearsonr(f_diff_array, N_swaps)
    plot_relation(f_diff_array, N_swaps,
                  "f_s_2qg_"+t1+"_"+meas_error, "decrement in fidelity", "# of SWAPS")
    print(f_s_corr)

    print("\n\t-- Correlation between the decrement in Prob. Success and # of SWAPS")

    ps_s_corr = pearsonr(ps_diff_array, N_swaps)
    plot_relation(ps_diff_array, N_swaps,
                  "ps_s_2qg_"+t1+"_"+meas_error, "decrement in fidelity", "# of SWAPS")
    print(ps_s_corr)


def two_q_gates_analysis(df_cl, t1, meas_error):

    mean_f = df_cl.mean_f
    N_two_qg = df_cl.N_two_qg
    prob_succs = df_cl.prob_succs

    print("\n\t-- Correlation between Fidelity and # of two-qubit gates")

    f_tqg_corr = pearsonr(mean_f, N_two_qg)
    plot_relation(mean_f, N_two_qg,
                  "f_2qg_"+t1+"_"+meas_error, "mean fidelity", "# of two-qubit gates")
    print(f_tqg_corr)

    print("\n\t-- Correlation between Probability of Success and # of two-qubit gates")

    ps_tqg_corr = pearsonr(prob_succs, N_two_qg)
    plot_relation(prob_succs, N_two_qg,
                  "ps_2qg_"+t1+"_"+meas_error, "Prob. success", "# of two-qubit gates")
    print(ps_tqg_corr)


def swap_proportion_analysis(df_cl, t1, meas_error):

    N_gates = df_cl.N_gates
    N_swaps = df_cl.N_swaps
    mean_f = df_cl.mean_f
    prob_succs = df_cl.prob_succs

    swaps_proportion = N_swaps/N_gates

    print("\n\t-- Correlation between Fidelity and the proportion of SWAPs")

    f_tqg_corr = pearsonr(mean_f, swaps_proportion)
    plot_relation(mean_f, swaps_proportion,
                  "f_sprop_"+t1+"_"+meas_error, "mean fidelity", "proportion of swaps")
    print(f_tqg_corr)

    print("\n\t-- Correlation between Probability of Success and the proportion of swaps")

    ps_tqg_corr = pearsonr(prob_succs, swaps_proportion)
    plot_relation(prob_succs, swaps_proportion,
                  "ps_sprop_"+t1+"_"+meas_error, "Prob. success", "proportion of swaps")
    print(ps_tqg_corr)


def fidelity_bar_plot(df_cl, t1, meas_error):

    df_nomapper = df_cl[df_cl["benchmark"] == "no"]
    df_rcmapper = df_cl[df_cl["benchmark"] == "minextendrc"]

    df_rcmapper.sort_values(by=["benchmark"])
    df_rcmapper.drop_duplicates(subset=["benchmark"], keep="first")
    df_nomapper.sort_values(by=["benchmark"])
    df_nomapper.drop_duplicates(subset=["benchmark"], keep="first")

    # Option 1
    ax = df_rcmapper.plot.bar(x="benchmark", y="mean_f")
    ax = df_nomapper.plot.bar(x="benchmark", y="mean_f")

    # # Option 2
    # x = list(range(1, 6))
    # ax = plt.subplot(111)
    # ax.bar(x-0.2, y, width=0.2, color='b', align='center')
    # ax.bar(x+0.2, k, width=0.2, color='r', align='center')

    # # Option 3
    # ax = plt.subplot(111)
    # ax.bar(df_nomapper["benchmark"], df_nomapper["mean_f"],
    #        width=0.2, color='b', align='center')
    # ax.bar(df_rcmapper["benchmark"], df_rcmapper["mean_f"],
    #        width=0.2, color='r', align='center')

    fig = ax.get_figure()
    fig.savefig("bar_plot_test.png")
    fig.clf()


def data_analysis(t1, meas_error):

    t1 = str(t1)
    meas_error = str(meas_error)
    N_gates = []
    N_swaps = []
    depth = []
    prob_succs = []
    mean_f = []
    q_vol = []
    N_two_qg = []
    mapper = []
    benchmark = []

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
            N_two_qg.append(two_q_gates[b_i[6]]+3*b_i[1])
            mapper.append(b_i[7])
            benchmark.append(b_i[6])

    data_frame = store_db_main_info(
        N_gates, N_two_qg, N_swaps, depth, prob_succs, mean_f, q_vol, mapper, benchmark)
    df_cl = clean_data_frame(data_frame)

    meas_error = meas_error.replace(".", "_")

    # general_results(df_cl, t1, meas_error)

    # two_q_gates_analysis(df_cl, t1, meas_error)

    # swap_proportion_analysis(df_cl, t1, meas_error)

    fidelity_bar_plot(df_cl, t1, meas_error)


data_analysis("3000", "0.005")
# data_analysis("1000", "0.005")
# data_analysis("3000", "0")
