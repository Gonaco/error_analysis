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

from math import ceil

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
                                 "mod5adder_127",
                                 "xor5_254",
                                 "hwb4_49"]


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


def extract_info_f_filter(db_path, t1, meas_error, f_min, f_max):

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # NO INITIAL PLACEMENT IS TO AVOID FOR NOW THE ERROR OF INITIAL PLACEMENTS AND THE NUMBER OF GATES TO AVOID THE ALGORITHM SYM6
    query = "SELECT DISTINCT HardwareBenchs.N_gates, HardwareBenchs.N_swaps, depth, prob_succs, mean_f, q_vol, Benchmarks.benchmark, mapper FROM SimulationsInfo LEFT JOIN HardwareBenchs ON algorithm=HardwareBenchs.id LEFT JOIN Results ON result=Results.id LEFT JOIN Experiments ON experiment=Experiments.id LEFT JOIN Benchmarks ON HardwareBenchs.benchmark=Benchmarks.id LEFT JOIN Configurations ON configuration=Configurations.id WHERE SimulationsInfo.t1 = {t1} AND meas_error = {meas_error} AND initial_placement='no' AND Benchmarks.N_gates < 3888 AND mean_f > {f_min} AND mean_f < {f_max};"
    cursor.execute(query.format(
        t1=t1, meas_error=meas_error, f_min=f_min, f_max=f_max))
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


def svm_regression(x, y, poly_order, exp):

    X = x
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1).ravel()

    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale', epsilon=0.001)

    # svr_lin = SVR(kernel='linear', C=1e3)
    # svr_poly = SVR(kernel='poly', C=1e3, degree=2)

    y_rbf = svr_rbf.fit(x, y).predict(x)
    # y_lin = svr_lin.fit(x, y).predict(x)
    # y_poly = svr_poly.fit(x, y).predict(x)

    z = np.polyfit(X, y_rbf, poly_order)
    if exp:
        z = np.polyfit(X, np.log(y_rbf), 1)
    f = np.poly1d(z)

    print("\nPolynomial function:")
    print(f)
    print("----------------------------\n")

    if exp:
        return np.exp(f(list(range(0, ceil(max(X))))))

    # return y_rbf, y_lin, y_poly
    # return y_rbf
    return f(list(range(0, ceil(max(X)))))
    # return f(list(np.arange(min(X), ceil(max(X)),0.01)))


def plot_relation(y, x, save_name, ylabel, xlabel, ax, linear=False, exp=False):
    # fig = plt.figure()
    ax.scatter(x, y)
    # fig.suptitle('test title', fontsize=20)

    # Fitting line (regression)
    # point, f = fit_polynomial(x, y, 1)
    # plt.plot(point, f, lw=2.5, c="k", label="fit line")

    # X_test, y_pred = linear_regression(x, y)
    # ax.plot(X_test, y_pred, linewidth=0.5, label="fit line (linear regression)",
    #         color='cornflowerblue', linestyle='dashed')

    # y_rbf, y_lin, y_poly = svm_regression(x, y)
    # plt.plot(x, y_rbf, color='navy', lw=3, label='RBF model')
    # plt.plot(x, y_lin, color='c', lw=3, label='Linear model')
    # plt.plot(x, y_poly, color='orange', lw=3, label='Polynomial model')

    y_poly = svm_regression(x, y, 1 if linear else 2, exp)
    # ax.plot(x, y_poly, lw=0.5, linestyle='dashed')
    ax.plot(list(range(0, ceil(max(x)))), y_poly, lw=1,
            label='Fitting line', linestyle='dashed')
    # ax.plot(list(np.arange(min(x), ceil(max(x)), 0.01)), y_poly, lw=1,
    #         label='Polynomial model', linestyle='dashed')

    # ax.legend()

    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.savefig(save_name)
    # plt.clf()


def clean_data_frame(data_frame):
    # I use Quantum Volume as the harder variable to be randomly repeated
    data_frame = data_frame.drop_duplicates(subset=['q_vol'], keep='first')
    return data_frame


def general_results(df_cl, t1, meas_error):

    print("\n\t-- Correlation between Fidelity and:")

    print("\n- # of Gates:")
    f_g_corr = pearsonr(df_cl.mean_f, df_cl.N_gates)
    plot_relation(df_cl.mean_f, df_cl.N_gates,
                  "f_g_"+t1+"_"+meas_error, "fidelity", "# of gates")
    print("\nPearson correlation:")
    print(f_g_corr)

    print("\n- # of Swaps:")
    f_s_corr = pearsonr(df_cl.mean_f, df_cl.N_swaps)
    plot_relation(df_cl.mean_f, df_cl.N_swaps,
                  "f_s_"+t1+"_"+meas_error, "fidelity", "# of swaps")
    print("\nPearson correlation:")
    print(f_s_corr)

    print("\n- Depth:")
    f_d_corr = pearsonr(df_cl.mean_f, df_cl.depth)
    plot_relation(df_cl.mean_f, df_cl.depth, "f_d_" +
                  t1+"_"+meas_error, "fidelity", "depth")
    print("\nPearson correlation:")
    print(f_d_corr)

    print("\n- Quantum Volume:")
    f_q_corr = pearsonr(df_cl.mean_f, df_cl.q_vol)
    plot_relation(df_cl.mean_f, df_cl.q_vol, "f_q_" +
                  t1+"_"+meas_error, "fidelity", "V_Q")
    print("\nPearson correlation:")
    print(f_q_corr)

    print("\n\n\t-- Correlation between Probability of Success and:")

    print("\n- # of Gates:")
    ps_g_corr = pearsonr(df_cl.prob_succs, df_cl.N_gates)
    plot_relation(df_cl.prob_succs, df_cl.N_gates,
                  "ps_g_"+t1+"_"+meas_error, "prob. success", "# of gates")
    print("\nPearson correlation:")
    print(ps_g_corr)

    print("\n- # of Swaps:")
    ps_s_corr = pearsonr(df_cl.prob_succs, df_cl.N_swaps)
    plot_relation(df_cl.prob_succs, df_cl.N_swaps,
                  "ps_s_"+t1+"_"+meas_error, "prob. success", "# of swaps")
    print("\nPearson correlation:")
    print(ps_s_corr)

    print("\n- Depth:")
    ps_d_corr = pearsonr(df_cl.prob_succs, df_cl.depth)
    plot_relation(df_cl.prob_succs, df_cl.depth,
                  "ps_d_"+t1+"_"+meas_error, "prob. success", "depth")
    print("\nPearson correlation:")
    print(ps_d_corr)

    print("\n- Quantum Volume:")
    ps_q_corr = pearsonr(df_cl.prob_succs, df_cl.q_vol)
    plot_relation(df_cl.prob_succs, df_cl.q_vol,
                  "ps_q_"+t1+"_"+meas_error, "prob. success", "V_Q")
    print("\nPearson correlation:")
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

    f_diff_array, N_swaps_f = fidelity_diff(df_cl)
    ps_diff_array, N_swaps_ps = prb_succs_diff(df_cl)

    print("\n\t-- Correlation between the decrement in Fidelity and # of SWAPS")

    f_s_corr = pearsonr(f_diff_array, N_swaps_f)
    plot_relation(f_diff_array, N_swaps_f,
                  "f_s_2qg_"+t1+"_"+meas_error, "decrement in fidelity", "# of SWAPS")
    print(f_s_corr)

    print("\n\t-- Correlation between the decrement in Prob. Success and # of SWAPS")

    ps_s_corr = pearsonr(ps_diff_array, N_swaps_ps)
    plot_relation(ps_diff_array, N_swaps_ps,
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

    df_nomapper = df_cl[df_cl["mapper"] == "no"]
    df_rcmapper = df_cl[df_cl["mapper"] == "minextendrc"]

    df_rcmapper = df_rcmapper.sort_values(by=["N_gates"])
    df_rcmapper = df_rcmapper.drop_duplicates(
        subset=["benchmark"], keep="first")
    df_nomapper = df_nomapper.sort_values(by=["N_gates"])
    df_nomapper = df_nomapper.drop_duplicates(
        subset=["benchmark"], keep="first")

    # Selecting the benchmarks
    df_nomapper = df_nomapper[df_nomapper["benchmark"].isin(
        benchmark_selection_corr_ps_f)]
    df_rcmapper = df_rcmapper[df_rcmapper["benchmark"].isin(
        benchmark_selection_corr_ps_f)]

    fig1, ax1 = plt.subplots()

    # # Option 1
    # df_nomapper.plot.bar(x="benchmark", y="mean_f", ax=ax1)
    # df_rcmapper.plot.bar(x="benchmark", y="mean_f", ax=ax1)

    # Option 2
    N_benchs = len(df_nomapper["mean_f"])
    x = list(range(1, N_benchs+1))
    x1 = [i-0.2 for i in range(1, N_benchs+1)]
    x2 = [i+0.2 for i in range(1, N_benchs+1)]

    ax1.bar(x1, df_nomapper["mean_f"], width=0.2, color=(
        0.2666, 0.4392, 0.5333), align='center', label="Before mapped")
    ax1.bar(x2, df_rcmapper["mean_f"], width=0.2, color=(
        0.3058, 0.7058, 0.9215), align='center', label="After mapped")
    ax1.legend()
    plt.xticks(x, df_rcmapper["benchmark"], rotation=0, fontsize=8)
    plt.ylabel("fidelity")

    # # Option 3
    # ax = plt.subplot(111)
    # ax.bar(df_nomapper["benchmark"], df_nomapper["mean_f"],
    #        width=0.2, color='b', align='center')
    # ax.bar(df_rcmapper["benchmark"], df_rcmapper["mean_f"],
    #        width=0.2, color='r', align='center')

    fig1.tight_layout()
    fig1.savefig("f_diff_bar_plot.png")
    fig1.savefig("f_diff_bar_plot_HQ.png", dpi=1000)
    fig1.savefig("f_diff_bar_plot.eps", dpi=1000)
    fig1.clf()


def fidelity_perctg(df_cl):

    f_perctg_array = []
    perctg_swaps = []

    for index, row in df_cl.iterrows():

        if row["N_swaps"] == 0:
            no_map_entr = row["mean_f"]
            d_b = row["depth"]
        else:
            # f_perctg_array.append((no_map_entr - row["mean_f"])/no_map_entr)
            # perctg_swaps.append(row["N_swaps"]/row["N_gates"])

            # Infidelity perc\entage
            f_perctg_array.append(-(row["mean_f"] -
                                    no_map_entr)/(1 - no_map_entr))
            # f_perctg_array.append(-(row["mean_f"] - no_map_entr)/no_map_entr)
            # f_perctg_array.append(-(row["mean_f"] - no_map_entr))

            # perctg_swaps.append(row["N_swaps"]/(row["N_gates"]-9*row["N_swaps"]))
            # perctg_swaps.append(row["N_swaps"])

            perctg_swaps.append((row["depth"] - d_b)/d_b)
            # perctg_swaps.append(row["depth"] - d_b)
            # perctg_swaps.append(row["depth"])
            # perctg_swaps.append(d_b)

    return f_perctg_array, perctg_swaps


def prb_succs_perctg(df_cl):

    ps_perctg_array = []
    perctg_swaps = []

    for index, row in df_cl.iterrows():

        if row["N_swaps"] == 0:
            no_map_entr = row["prob_succs"]
        else:
            ps_perctg_array.append(
                (no_map_entr - row["prob_succs"])/no_map_entr)
            perctg_swaps.append(row["N_swaps"]/row["N_gates"])

    return ps_perctg_array, perctg_swaps


def diff_f_ps_swap_percentage(df_cl, t1, meas_error, axf, axps):

    f_perctg_array, perctg_swaps_f = fidelity_perctg(df_cl)
    ps_perctg_array, perctg_swaps_ps = prb_succs_perctg(df_cl)

    print("\n\t-- Correlation between the percentage of decrement in Fidelity and percentage of SWAPS")

    f_s_corr = pearsonr(f_perctg_array, perctg_swaps_f)
    plot_relation(f_perctg_array, perctg_swaps_f,
                  "f_swap_percentage_"+t1+"_"+meas_error, "percentage of decrement in fidelity", "percentage of SWAPS", axf)
    print(f_s_corr)

    print("\n\t-- Correlation between the percentage of decrement in Prob. Success and percentage of SWAPS")

    ps_s_corr = pearsonr(ps_perctg_array, perctg_swaps_ps)
    plot_relation(ps_perctg_array, perctg_swaps_ps,
                  "ps_swap_percentage_"+t1+"_"+meas_error, "percentage of decrement in Probability of success", "percentage of SWAPS", axps)
    print(ps_s_corr)


def minus_infid_percentage_depth_before(df_cl, t1, meas_error, axf, axps):

    print("\n\t-- Correlation between the percentage of decrement in Infidelity and depth before being mapped")

    infid_perctg_array = []
    depth_before = []

    for index, row in df_cl.iterrows():

        if row["N_swaps"] == 0:
            no_map_entr = row["mean_f"]
            d_b = row["depth"]
        else:
            # Infidelity perc\entage
            infid_perctg_array.append(-(row["mean_f"] -
                                        no_map_entr)/(1 - no_map_entr))

            depth_before.append(d_b)

    f_s_corr = pearsonr(infid_perctg_array, depth_before)
    plot_relation(infid_perctg_array, depth_before,
                  "f_swap_percentage_"+t1+"_"+meas_error, "percentage of decrement in fidelity", "percentage of SWAPS", axf)
    print(f_s_corr)


def f_ps_correlation(df_cl, t1, meas_error, ax):

    f = df_cl["mean_f"]
    ps = df_cl["prob_succs"]

    print("\n\t-- Correlation between the Fidelity and Probability of Success")

    f_ps_corr = pearsonr(ps, f)
    print(f_ps_corr)

    ax.scatter(f, ps)

    X = f
    x = np.array(f).reshape(-1, 1)
    y = np.array(ps).reshape(-1, 1).ravel()

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale', epsilon=0.001)

    y_rbf = svr_rbf.fit(x, y).predict(x)

    z = np.polyfit(X, np.exp(y_rbf), 1)
    f_poly = np.poly1d(z)

    print("\nPolynomial function:")
    print(f_poly)
    print("----------------------------\n")

    y_poly = np.log(f_poly(list(np.arange(min(X), ceil(max(X)), 0.01))))

    ax.plot(list(np.arange(min(X), ceil(max(X)), 0.01)), y_poly, lw=1,
            label='Fitting line', linestyle='dashed')

    f_ps_corr = pearsonr(np.exp(ps), f)
    print(f_ps_corr)


def f_ps_metrics_correlation(df_cl, t1, meas_error, axarr1, axarr2):

    print("\n\t-- Correlation between Fidelity and:")

    print("\n- # of Gates:")
    f_g_corr = pearsonr(df_cl.mean_f, df_cl.N_gates)
    print(f_g_corr)
    plot_relation(df_cl.mean_f, df_cl.N_gates,
                  "f_g_"+t1+"_"+meas_error, "fidelity", "# of gates", axarr1[0, 0], exp=True if t1 == "3000" else False)
    axarr1[0, 0].set_ylabel("fidelity")
    axarr1[0, 0].set_xlabel("# of gates")
    axarr1[0, 0].set_ylim(0, 1)

    f_g_corr = pearsonr(np.log(df_cl.mean_f), df_cl.N_gates)
    print(f_g_corr)

    print("\n- # of two-qubit gates:")
    f_s_corr = pearsonr(df_cl.mean_f, df_cl.N_two_qg)
    print(f_s_corr)
    plot_relation(df_cl.mean_f, df_cl.N_two_qg,
                  "f_s_"+t1+"_"+meas_error, "fidelity", "# of two-qubit gates", axarr1[0, 1], exp=True if t1 == "3000" else False)
    axarr1[0, 1].set_ylabel("fidelity")
    axarr1[0, 1].set_xlabel("# of two-qubit gates")
    axarr1[0, 1].set_ylim(0, 1)
    axarr1[0, 1].legend(labels=["Fitting line", "Fitting line",
                                "t_d 30 µs", "t_d 10 µs"], fontsize=8, frameon=True)

    f_s_corr = pearsonr(np.log(df_cl.mean_f), df_cl.N_two_qg)
    print(f_s_corr)

    print("\n- Depth:")
    f_d_corr = pearsonr(df_cl.mean_f, df_cl.depth)
    print(f_d_corr)
    plot_relation(df_cl.mean_f, df_cl.depth, "f_d_" +
                  t1+"_"+meas_error, "fidelity", "depth", axarr1[1, 0], exp=True if t1 == "3000" else False)
    axarr1[1, 0].set_ylabel("fidelity")
    axarr1[1, 0].set_xlabel("depth")
    axarr1[1, 0].set_ylim(0, 1)

    f_d_corr = pearsonr(np.log(df_cl.mean_f), df_cl.depth)
    print(f_d_corr)

    print("\n- Quantum Volume:")
    f_q_corr = pearsonr(df_cl.mean_f, df_cl.q_vol)
    print(f_q_corr)
    plot_relation(df_cl.mean_f, df_cl.q_vol, "f_q_" +
                  t1+"_"+meas_error, "fidelity", "V_Q", axarr1[1, 1], exp=True if t1 == "3000" else False)
    axarr1[1, 1].set_ylabel("fidelity")
    axarr1[1, 1].set_xlabel("Quantum Volume")
    axarr1[1, 1].set_ylim(0, 1)

    f_q_corr = pearsonr(np.log(df_cl.mean_f), df_cl.q_vol)
    print(f_q_corr)

    print("\n\n\t-- Correlation between Probability of Success and:")

    print("\n- # of Gates:")
    ps_g_corr = pearsonr(df_cl.prob_succs, df_cl.N_gates)
    print(ps_g_corr)
    plot_relation(df_cl.prob_succs, df_cl.N_gates,
                  "ps_g_"+t1+"_"+meas_error, "prob. success", "# of gates", axarr2[0, 0], True, True)
    axarr2[0, 0].set_ylabel("prob. of success")
    axarr2[0, 0].set_xlabel("# of gates")
    axarr2[0, 0].set_ylim(0, 1)

    ps_g_corr = pearsonr(np.log(df_cl.prob_succs), df_cl.N_gates)
    print(ps_g_corr)

    print("\n- # of two-qubit gates:")
    ps_s_corr = pearsonr(df_cl.prob_succs, df_cl.N_two_qg)
    print(ps_s_corr)
    plot_relation(df_cl.prob_succs, df_cl.N_two_qg,
                  "ps_s_"+t1+"_"+meas_error, "prob. success", "# of -qubit gates", axarr2[0, 1], True, True)
    axarr2[0, 1].set_ylabel("prob. of success")
    axarr2[0, 1].set_xlabel("# of two-qubit gates")
    axarr2[0, 1].set_ylim(0, 1)
    axarr2[0, 1].legend(labels=["Fitting line", "Fitting line",
                                "t_d 30 µs", "t_d 10 µs"], fontsize=8, frameon=True)

    ps_s_corr = pearsonr(np.log(df_cl.prob_succs), df_cl.N_two_qg)
    print(ps_s_corr)

    print("\n- Depth:")
    ps_d_corr = pearsonr(df_cl.prob_succs, df_cl.depth)
    print(ps_d_corr)
    plot_relation(df_cl.prob_succs, df_cl.depth,
                  "ps_d_"+t1+"_"+meas_error, "prob. success", "depth", axarr2[1, 0], True, True)
    axarr2[1, 0].set_ylabel("prob. of success")
    axarr2[1, 0].set_xlabel("depth")
    axarr2[1, 0].set_ylim(0, 1)

    ps_d_corr = pearsonr(np.log(df_cl.prob_succs), df_cl.depth)
    print(ps_d_corr)

    print("\n- Quantum Volume:")
    ps_q_corr = pearsonr(df_cl.prob_succs, df_cl.q_vol)
    print(ps_q_corr)
    plot_relation(df_cl.prob_succs, df_cl.q_vol,
                  "ps_q_"+t1+"_"+meas_error, "prob. success", "V_Q", axarr2[1, 1], True, True)
    axarr2[1, 1].set_ylabel("prob. of success")
    axarr2[1, 1].set_xlabel("Quantum Volume")
    axarr2[1, 1].set_ylim(0, 1)

    ps_q_corr = pearsonr(np.log(df_cl.prob_succs), df_cl.q_vol)
    print(ps_q_corr)


def f_s_d_3d_plot(ax, fidelity, N_swaps, depth):

    ax.scatter(N_swaps, depth, fidelity)

    ax.set_xlabel('# swaps')
    ax.set_ylabel('depth')
    ax.set_zlabel('fidelity')


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

    # fidelity_bar_plot(df_cl, t1, meas_error)
    diff_f_ps_swap_percentage(df_cl, t1, meas_error)


def thesis_bar_plot():

    param = [["3000", "0.005"]]

    for p in param:

        t1 = p[0]
        meas_error = p[1]
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

        meas_error_ = meas_error.replace(".", "_")

        fidelity_bar_plot(df_cl, t1, meas_error_)


def thesis_mapping_effect():

    param = [["3000", "0.005"]]

    # param = [["1000", "0.005"]]
    figf, axf = plt.subplots()
    plt.xlabel("Benchmarks")
    # plt.ylabel("-1x infidelity difference percentage")
    plt.ylabel("fidelity")
    figdiff, axdiff = plt.subplots()
    plt.xlabel("Benchmarks")
    plt.ylabel("fidelity difference percentage")

    for p in param:

        t1 = p[0]
        meas_error = p[1]
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

            bench_info = extract_info(db_path, t1, meas_error)
            # bench_info = extract_info_f_filter(
            #     db_path, t1, meas_error, 0.5, 1)
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

        meas_error_ = meas_error.replace(".", "_")

        error_metric = []
        circuit_metric = []
        error_metric_no_mapped = []

        table_print = "| {bench} | {depth} |"

        error_metric_diff_up = []
        error_metric_diff_low = []
        error_metric_diff = []
        circuit_metric_diff = []
        benchmark_prev = ""

        for index, row in df_cl.iterrows():

            # print(table_print.format(
            #     bench=row["benchmark"], depth=row["depth"]))

            if row["N_swaps"] == 0:
                no_map_entr = row["mean_f"]
                d_b = row["depth"]
                benchmark = row["benchmark"]
            else:
                # error_metric.append((no_map_entr - row["mean_f"])/no_map_entr)
                # circuit_metric.append(row["N_swaps"]/row["N_gates"])

                # Infidelity perc\entage
                # error_metric.append(-(row["mean_f"] -
                #                       no_map_entr)/(1 - no_map_entr))
                # error_metric.append(-(row["mean_f"] -
                #                       no_map_entr)/no_map_entr)
                # error_metric.append(-(row["mean_f"] - no_map_entr))
                # circuit_metric.append(row["N_swaps"]/(row["N_gates"]-9*row["N_swaps"]))
                # circuit_metric.append(row["N_swaps"])

                # circuit_metric.append((row["depth"] - d_b)/d_b)
                # circuit_metric.append(row["depth"] - d_b)
                # circuit_metric.append(row["depth"])

                error_metric.append(row["mean_f"])
                circuit_metric.append(d_b)

                error_metric_no_mapped.append(no_map_entr)

                if benchmark != benchmark_prev:

                    benchmark_prev = benchmark

                    if error_metric_diff:
                        error_metric_diff_up.append(max(error_metric_diff))
                        error_metric_diff_low.append(min(error_metric_diff))

                    circuit_metric_diff.append(d_b)

                    error_metric_diff = []

                error_metric_diff.append(-(row["mean_f"] -
                                           no_map_entr)/no_map_entr)

        error_metric_diff_up.append(max(error_metric_diff))
        error_metric_diff_low.append(min(error_metric_diff))

        print("\n\t-- Correlation between the percentage of decrement in Fidelity and percentage of SWAPS")

        f_s_corr = pearsonr(error_metric, circuit_metric)
        axf.scatter(circuit_metric, error_metric_no_mapped,
                    color=(0.2666, 0.4392, 0.5333), label="Before mapped")
        axf.scatter(circuit_metric, error_metric,
                    color=(0.3058, 0.7058, 0.9215), label="After mapped")
        axf.legend(frameon=True)
        axf.set_xticklabels([])

        axdiff.scatter(circuit_metric_diff, error_metric_diff_up,
                       color=(0.92, 0.36, 0.35), label="Upper bound")
        axdiff.scatter(circuit_metric_diff, error_metric_diff_low,
                       color=(0.11, 0.4, 0.6), label="Lower bound")
        axdiff.legend(frameon=True)

        # X = circuit_metric_diff
        # x = np.array(circuit_metric_diff).reshape(-1, 1)
        # y = np.array(error_metric_diff_up).reshape(-1, 1).ravel()

        # svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale', epsilon=0.001)

        # y_rbf = svr_rbf.fit(x, y).predict(x)

        # z = np.polyfit(X, np.exp(y_rbf), 1)
        # f_poly = np.poly1d(z)

        # print("\nPolynomial function:")
        # print(f_poly)
        # print("----------------------------\n")

        # y_poly = np.log(f_poly(list(np.arange(min(X), ceil(max(X)), 0.01))))

        # axdiff.plot(list(np.arange(min(X), ceil(max(X)), 0.01)), y_poly, lw=1,
        #             label='Fitting line', linestyle='dashed', color=(0.92, 0.36, 0.35))

        # X = circuit_metric_diff
        # x = np.array(circuit_metric_diff).reshape(-1, 1)
        # y = np.array(error_metric_diff_low).reshape(-1, 1).ravel()

        # svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale', epsilon=0.001)

        # y_rbf = svr_rbf.fit(x, y).predict(x)

        # z = np.polyfit(X, np.exp(y_rbf), 1)
        # f_poly = np.poly1d(z)

        # print("\nPolynomial function:")
        # print(f_poly)
        # print("----------------------------\n")

        # y_poly = np.log(f_poly(list(np.arange(min(X), ceil(max(X)), 0.01))))

        # axdiff.plot(list(np.arange(min(X), ceil(max(X)), 0.01)), y_poly, lw=1,
        #             label='Fitting line', linestyle='dashed', color=(0.7, 0.7058, 0.13))

        axdiff.set_xticklabels([])
        # axf.bar(circuit_metric, error_metric)
        print(f_s_corr)

    figf.tight_layout()
    figf.savefig("mapping_effect_"+t1+".png")
    figf.savefig("mapping_effect_"+t1+"_HQ.png", dpi=1000)
    figf.savefig("mapping_effect_"+t1+".eps", dpi=1000)
    figf.clf()

    figdiff.tight_layout()
    figdiff.savefig("mapping_effect_diff_"+t1+".png")
    figdiff.savefig("mapping_effect_diff_"+t1+"_HQ.png", dpi=1000)
    figdiff.savefig("mapping_effect_diff_"+t1+".eps", dpi=1000)
    figdiff.clf()


def thesis_f_ps_corr_plot():

    param = [["3000", "0.005"], ["1000", "0.005"]]

    figfps, axfps = plt.subplots()
    plt.xlabel("fidelity")
    plt.ylabel("prob. of success")

    for p in param:

        t1 = p[0]
        meas_error = p[1]
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
            # bench_info = extract_info_f_filter(
            #     db_path, t1, meas_error, 0.5, 1)
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

        meas_error_ = meas_error.replace(".", "_")

        f_ps_correlation(df_cl, t1, meas_error, axfps)

    axfps.legend(labels=["Fitting line", "Fitting line",
                         "t_d 30 µs", "t_d 10 µs"], fontsize=8, frameon=True)

    axfps.set_ylim(0, 1)
    axfps.plot(axfps.get_xlim(), axfps.get_ylim(), color="gray", ls=":",
               lw=1, label='Prob. succ = Fidelity')

    figfps.tight_layout()
    figfps.savefig("f_ps_correlation.png")
    figfps.savefig("f_ps_correlation_HQ.png", dpi=1000)
    figfps.savefig("f_ps_correlation.eps", dpi=1000)
    figfps.clf()


def thesis_f_ps_corr_plot_filt():

    param = [["3000", "0.005"], ["1000", "0.005"]]

    figfps, axfps = plt.subplots()
    plt.xlabel("fidelity")
    plt.ylabel("prob. of success")

    for p in param:

        t1 = p[0]
        meas_error = p[1]
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
            # bench_info = extract_info(db_path, t1, meas_error)
            bench_info = extract_info_f_filter(
                db_path, t1, meas_error, 0.5, 1)
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

        meas_error_ = meas_error.replace(".", "_")

        f_ps_correlation(df_cl, t1, meas_error, axfps)

    axfps.legend(labels=["Fitting line", "Fitting line",
                         "t_d 30 µs", "t_d 10 µs"], fontsize=8, frameon=True)

    axfps.set_ylim(0, 1)
    axfps.plot(axfps.get_xlim(), axfps.get_ylim(), color="gray", ls=":",
               lw=1, label='Prob. succ = Fidelity')

    figfps.tight_layout()
    figfps.savefig("f_ps_correlation.png")
    figfps.savefig("f_ps_correlation_HQ.png", dpi=1000)
    figfps.savefig("f_ps_correlation.eps", dpi=1000)
    figfps.clf()


def thesis_f_ps_metrics_correlation():

    param = [["3000", "0.005"], ["1000", "0.005"]]

    figmf, axarrf = plt.subplots(2, 2)
    figmps, axarrps = plt.subplots(2, 2)

    for p in param:

        t1 = p[0]
        meas_error = p[1]
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

            bench_info = extract_info(db_path, t1, meas_error)
            # bench_info = extract_info_f_filter(
            #     db_path, t1, meas_error, 0.5, 1)
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

        meas_error_ = meas_error.replace(".", "_")

        f_ps_metrics_correlation(df_cl, t1, meas_error, axarrf, axarrps)

    # figmf.legend("Fitting line", fontsize=8)
    figmf.tight_layout()
    figmf.savefig("f_metrics_correlation.png")
    figmf.savefig("f_metrics_correlation_HQ.png", dpi=1000)
    figmf.savefig("f_metrics_correlation.eps", dpi=1000)
    figmf.clf()

    # figmps.legend("Fitting line", fontsize=8)
    figmps.tight_layout()
    figmps.savefig("ps_metrics_correlation.png")
    figmps.savefig("ps_metrics_correlation_HQ.png", dpi=1000)
    figmps.savefig("ps_metrics_correlation.eps", dpi=1000)
    figmps.clf()


def thesis_f_ps_metrics_correlation_filt():

    param = [["3000", "0.005"], ["1000", "0.005"]]

    figmf, axarrf = plt.subplots(2, 2)
    figmps, axarrps = plt.subplots(2, 2)

    for p in param:

        t1 = p[0]
        meas_error = p[1]
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

            # bench_info = extract_info(db_path, t1, meas_error)
            bench_info = extract_info_f_filter(
                db_path, t1, meas_error, 0.5, 1)
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

        meas_error_ = meas_error.replace(".", "_")

        f_ps_metrics_correlation(df_cl, t1, meas_error, axarrf, axarrps)

    # figmf.legend("Fitting line", fontsize=8)
    figmf.tight_layout()
    figmf.savefig("f_metrics_correlation_filtered.png")
    figmf.savefig("f_metrics_correlation_filtered_HQ.png", dpi=1000)
    figmf.savefig("f_metrics_correlation_filtered.eps", dpi=1000)
    figmf.clf()

    # figmps.legend("Fitting line", fontsize=8)
    figmps.tight_layout()
    figmps.savefig("ps_metrics_correlation_filtered.png")
    figmps.savefig("ps_metrics_correlation_filtered_HQ.png", dpi=1000)
    figmps.savefig("ps_metrics_correlation_filtered.eps", dpi=1000)
    figmps.clf()


def thesis_f_swaps_depth_correlation():

    param = [["3000", "0.005"], ["1000", "0.005"]]

    figfsd, axarrfsd = plt.subplots(2, 2)

    for p in param:

        t1 = p[0]
        meas_error = p[1]
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

            # bench_info = extract_info(db_path, t1, meas_error)
            bench_info = extract_info_f_filter(
                db_path, t1, meas_error, 0.5, 1)
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

        meas_error_ = meas_error.replace(".", "_")

        f_s_d_3d_plot(axarrfsd)

    # figmf.legend("Fitting line", fontsize=8)
    figfsd.tight_layout()
    figfsd.savefig("f_swaps_depth_3d.png")
    figfsd.savefig("f_swaps_depth_3d_HQ.png", dpi=1000)
    figfsd.savefig("f_swaps_depth_3d.eps", dpi=1000)
    figfsd.clf()

    return


thesis_bar_plot()
thesis_mapping_effect()
thesis_f_ps_corr_plot()
thesis_f_ps_metrics_correlation()
thesis_f_swaps_depth_correlation()
