import numpy as np
import os
import atexit
import re

# Regular Expresion for extracting quantum state
# https://regex101.com/r/VBkD3d/1

# QX ##########################################################################

import qxelarator

# GRAPHS ######################################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from palettable.mycarta import Cube1_20

# QUANTUMSIM ##################################################################

from quantumsim.circuit import Circuit
from quantumsim.circuit import uniform_noisy_sampler
import quantumsim.sparsedm as sparsedm

# Momentary gates added
from quantumsim.circuit import CNOT as cnot
from quantumsim.circuit import Hadamard as h
from quantumsim.circuit import RotateEuler as RotateEuler
from quantumsim.circuit import ResetGate as ResetGate

# CONSTANTS ###################################################################

INIT_QST_FILE = "init_state.qst"

# FILE TREATMENT FUNCTIONS ####################################################


def delcopy(cp):

    os.remove(cp)


def graph(N_qubits, matrix, file_name):
    """Draw a graph for the all input analysis
    """

    fig = plt.figure(figsize=(7, 7))

    # First graph (3D histogram)
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    # Tableau Colors
    # ax.set_color_cycle(Tableau_20.mpl_colors)

    # Background color
    ax.set_facecolor("white")

    # Set perspective
    ax.view_init(35, -45)

    x = np.arange(2**N_qubits)
    y = np.arange(2**N_qubits)
    xpos, ypos = np.meshgrid(x, y)

    axis = [format(i, "0"+str(N_qubits)+"b") for i in range(2**N_qubits)]

    xpos = xpos.flatten()   # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(2**(2*N_qubits))

    dx = 0.75 * np.ones_like(zpos)
    dy = dx.copy()
    dz = matrix.flatten()

    ratio = int(20/(2**N_qubits)) if int(20/(2**N_qubits)) != 0 else 1
    end = 2**N_qubits * ratio
    # cs = Tableau_20.mpl_colors[:8] * 2**N_qubits
    cs_y = Cube1_20.mpl_colors[:end:ratio] * 2**N_qubits
    order = [i for i in range(2**N_qubits)] * 2**N_qubits
    cs_x = [x for _, x in sorted(zip(order, cs_y))]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
             color=cs_x, shade=False, edgecolor="k")

    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
    #          cmap=Cube1_20.mpl_colormap, edgecolor='b')

    # sh()
    ax.w_xaxis.set_ticklabels(axis)
    ax.w_yaxis.set_ticklabels(axis)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()

    ax.set_xlabel("Actual Results")
    ax.set_ylabel("Expected Results (Correct)")
    ax.set_zlabel("Prob. Success")

    fig.tight_layout()

    plt.savefig(file_name+"_tomography_graph")

    # Second plot. Heatmap

    fig2 = plt.figure(figsize=(7, 7))
    ax2 = fig2.add_subplot(111)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    im = ax2.imshow(matrix, cmap="jet")
    # im = ax2.imshow(matrix, cmap=Cube1_20.mpl_colormap)

    ax2.set_xticks(np.arange(2**N_qubits))
    ax2.set_yticks(np.arange(2**N_qubits))
    ax2.set_xticklabels(axis)
    ax2.set_yticklabels(axis)

    for i in range(2**N_qubits):
        for j in range(2**N_qubits):
            text = ax2.text(j, i, round(matrix[i, j], 2),
                            ha="center", va="center", color="w")

    ax2.set_xlabel("Expected Results (Correct)")
    ax2.set_ylabel("Actual Results")
    ax2.set_title("Prob. Success")

    plt.colorbar(im, cax=cax)

    # plt.show()
    fig2.tight_layout()

    plt.savefig(file_name+"_heatmap")


def just_heatmap(N_qubits, matrix, file_name):

    axis = [format(i, "0"+str(N_qubits)+"b") for i in range(2**N_qubits)]

    fig2 = plt.figure(figsize=(7, 7))
    ax2 = fig2.add_subplot(111)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    im = ax2.imshow(matrix, cmap="jet")
    # im = ax2.imshow(matrix, cmap=Cube1_20.mpl_colormap)

    ax2.set_xticks(np.arange(2**N_qubits))
    ax2.set_yticks(np.arange(2**N_qubits))
    ax2.set_xticklabels(axis)
    ax2.set_yticklabels(axis)

    for i in range(2**N_qubits):
        for j in range(2**N_qubits):
            text = ax2.text(j, i, round(matrix[i, j], 2),
                            ha="center", va="center", color="w")

    ax2.set_xlabel("Expected Results (Correct)")
    ax2.set_ylabel("Actual Results")
    ax2.set_title("Prob. Success")

    plt.colorbar(im, cax=cax)

    # plt.show()
    fig2.tight_layout()

    plt.savefig(file_name+"_heatmap")


# Classes #################################################################

class QASMReader(object):
    '''An object able to read and store information from a QASM file'''

    def __init__(self, file_path):

        self.file_path = file_path
        with open(file_path, "r") as i:
            self.data = i.readlines()

        self.N_qubits = 0
        self.depth = 0
        self.N_gates = 0
        self.N_swaps = 0

    def save(self, output_path):

        with open(output_path, "w") as o:
            o.writelines(self.data)

    def search(regex, line):

        match = re.search(regex, line)
        if match:
            return match

    def isQasm(filename):

        if "qasm" in filename:

            return True

    def num_qubitsChecker(line, num_qubits):

        correction = line

        if "qubits" in line:
            correction = "qubits {num_qubits}\n".format(num_qubits=num_qubits)

            return correction
        else:

            return False

    def bigQubitNum(line, biggest_number):

        match = re.findall(r"q\[?(\d+)\]?", line)

        # print("\nmatch:")
        # print(match)

        # print("\nmatch2int:")
        # print(list(map(int, match)))

        if match:
            '''The biggest qubit number between the biggest of all the qubit numbers in a line and the previous biggest number'''
            biggest_number = max(max(list(map(int, match))), biggest_number)

        return biggest_number

    def algNameChecker(line):

        correction = doubleNameChecker(line)

        return numberNameChecker(correction)

    def doubleNameChecker(line):

        correction = line
        before = r"(\..+)\..+$"
        after = r"\1"

        c = re.sub(before, after, correction)

        if c:
            correction = c

        return correction

    def numberNameChecker(line):

        correction = line
        before = r"^\.(\d.+)"
        after = r".kernel_\1"

        c = re.sub(before, after, correction)

        if c:
            correction = c

        return correction

    def versionChecker(corrected):

        # isVersion = False

        for c in range(5):          # Look in the first 5 lines for the version line

            if "version" in corrected[c]:
                # isVersion = True
                # break
                return corrected

        # if not isVersion:
        #     corrected.insert(0, "version 2.0")

        # print(corrected[0])

        corrected.insert(0, "version 2.0\n\n")
        # corrected.insert(0, "version 1.0\n\n")

        return corrected

    def parenthesisChecker(line):

        correction = line
        before = r"q(\d+)"
        after = r"q[\1]"

        c = re.sub(before, after, correction)

        if c:
            correction = c

        return correction

    def rotationGatesChecker(line):

        correction = line
        # There could be two different kind of rotations, + and - 90
        before = [r"r([xy])(\d+) (q\[?\d+\]?)", r"([xy])m(\d+) (q\[?\d+\]?)"]
        after = [r"r\1 \3, \2", r"r\1 \3, -\2"]

        for i, b in enumerate(before):

            c = re.sub(b, after[i], correction)

            if c:
                correction = c

        return correction

    def check_cQasm(filename):

        print("\n\ncheck_cQasm {filename}\n".format(filename=filename))

        if not isQasm(filename):

            print(
                "\nERROR. The file is not a QASM file. Please, use a cQASM file as input\n")

            return

        backup = ""
        corrected = []
        biggest_number = 0

        backup = self.data

        for line in data:

            line = algNameChecker(line)

            line = parenthesisChecker(line)

            line = rotationGatesChecker(line)

            biggest_number = bigQubitNum(line, biggest_number)

            corrected.append(line)

        corrected = versionChecker(corrected)

        try:

            for idx, line in enumerate(corrected):
                isLine = num_qubitsChecker(line, biggest_number+1)
                if isLine:
                    corrected[idx] = isLine
                    break

            f.writelines(corrected)

        except:

            f.writelines(backup)
            raise

        return

    def searchDepth(self, line):

        self.N_qubits = int(self.search("# Total depth: (\d+)", line)[1])

    def getDepth(self):

        return self.depth

    def searchN_qubits(self, line):

        self.N_qubits = int(self.search("^qubits (\d*)", line)[1])

    def getN_Qubits(self):

        return self.N_qubits

    def searchN_gates(self, line):

        self.N_qubits = int(self.search(
            "# Total no. of quantum gates: (\d+)", line)[1])

    def getN_Gates(self):

        return self.N_gates

    def extractInfo(self, line):

        self.searchN_qubits(line)
        self.searchDepth(line)
        self.searchN_gates(lines)

    def add2qasm(self, before, after):
        """Look for some regular expression in a file (before)
        and add something new after it in a copy of this file
        """

        for idx, line in enumerate(self.data):

            extractInfo(line)

            if re.search(before, line):
                self.data[idx] = line+"\n"+after

    def add_error_model(errprob):

        error_model = "error_model depolarizing_channel, " + str(errprob)
        self.add2qasm("^qubits \d+", error_model)

    def addinit(self):
        """
        """

        init = '\n.init\n    load_state "'+INIT_QST_FILE+'"\n'
        self.add2qasm("^qubits \d+", init)

        self.add_measurement()

    def add_measurement(self):
        """Appending the measurement to the end of the qasm file
        """
        m_string = "   measure q"
        measurements = ["\n"]

        for q in range(self.N_qubits):
            measurements.append(m_string+str(q)+"\n")

        self.data.append(measurements)

    def cleaning_qwaits(self):

        for idx, line in enumerate(self.data):

            match = re.findall(r"qwait \d+", line)
            # print(match)

            if match:
                self.data.pop(idx)


class Analysis(object):

    '''Object that runs the error analysis of some benchmarks'''

    def __init__(self, algs_path, log_path, h5_path, config_file_path, scheduler, mapper, initial_placement, output_dir_name, init_type, error, simulator):

        self.algs_path = algs_path
        self.log_path = log_path
        self.h5_path = h5_path
        self.connection = sqlite3.connect("error_analysis.db")
        self.cursor = connection.cursor()
        self.config_file_path = config_file_path
        self.scheduler = scheduler
        self.mapper = mapper

        self.initial_placement = initial_placement
        self.output_dir_name = output_dir_name
        self.init_type = init_type
        self.error = error
        self.simulator = simulator

    def save_in_db(cursor, algorithm, N_sim, init_type, scheduler, mapper, initial_placement, error_rate, conf_file, prob_succs, mean_f, q_vol, exper_id, simulator):

        if init_type == 0:
            init_type = "all_states"
        elif init_type == 1:
            init_type = "ground_state"
        else:
            print("Init type ERROR. The init_type is not either 1 nor 0")
            raise Exception("Init_TypeError")

        if simulator:
            simulator = "quantumsim"
        else:
            simulator = "qx"

        format_str = "INSERT INTO Results (algorithm, N_sim, init_type, scheduler, mapper, initial_placement, error_rate, conf_file, prob_succs, mean_f, q_vol, simulator, exper_id) VALUES ('{algorithm}', {N_sim}, '{init_type}', '{scheduler}', '{mapper}', '{initial_placement}', {error_rate}, '{conf_file}', {prob_succs}, {mean_f}, {q_vol}, {simulator}, {exper_id});"

        cursor.execute(format_str.format(algorithm=algorithm, N_sim=N_sim, init_type=init_type,
                                         scheduler=scheduler, mapper=mapper, initial_placement=initial_placement, error_rate=error_rate, conf_file=conf_file, prob_succs=prob_succs, mean_f=mean_f, q_vol=q_vol, exper_id=exper_id, simulator=simulator))

    def db_init_query(self):

        new_experiment_query = "INSERT INTO Experiments (date, tom_mtrx_path, log_path) VALUES (datetime('now'),'" + \
            self.h5_path+"', '"+self.log_path+"');"
        self.cursor.execute(new_experiment_query)
        self.connection.commit()

        self.cursor.execute("SELECT last_insert_rowid() FROM Experiments;")
        experiment_id = cursor.fetchone()[0]

        return experiment_id

    def db_interruption_query(self):

        self.connection.commit()
        self.connection.close()

    def db_final_query(self):

        self.cursor.execute("UPDATE Experiments SET fail = 0 WHERE id =" +
                            str(experiment_id)+";")
        self.connection.commit()
        self.connection.close()

    def analysis(self):

        with h5py.File(h5_path, "w") as h5f:

            experiment_id = self.db_init_query()

            try:

                for alg_path in algs_path:

                    self.descripBench = DescripBench(alg_path)
                    self.descripBench.compile(config_file_path)
                    self.simBench = SimBench(qasm_file_path)

            except:
                self.db_interruption_query()
                raise

        self.db_final_query()


class DescripBench(object):

    '''Object for taking care of the OpenQL benchmarks'''

    def __init__(self, file_path):

        self.openql = importlib.util.spec_from_file_location(
            alg_path, filename)
        self.openql_comp = importlib.util.module_from_spec(openql)

    def compile(self, config_file_path, scheduler, mapper, ini):

        self.openql.loader.exec_module(openql_comp)

        try:
            openql_comp.circuit(
                config_file_path, scheduler, mapper, initial_placement, output_dir_name)


class SimBench(object):

    '''Class for simulating the Benchmark'''

    def __init__(self, qasm_file_path, N_exp=1000):

        self.qasm_file_path = qasm_file_path
        self.cp = "."+qasm_file_path+"~"

        self.N_exp = N_exp
        self.success_registry = []  # Matrix storing the success
        self.fidelity_registry = []  # Matrix storing the fidelity
        self.total_meas_err = 0

        # Initializing qasm copy
        try:
            self.N_qubits = addinit(qasm_file_path, self.cp)

        except FileNotFoundError:
            print(
                "\nThe QASM file does not exist or the path is incorrect." +
                "\nThe Benchmark cannot be created\n")
            raise

        # atexit.register(delcopy, cp="."+qasm_file_path+"~")
        # atexit.register(delcopy, cp="."+qasm_file_path.replace(
        #     ".qasm", "_error.qasm")+"~")

        self.tomography_matrix = np.zeros((2**self.N_qubits, 2**self.N_qubits))

        # Initializing quantumsim
        try:
            self.qsimc = __import__(
                qasm_file_path.replace(".qasm", "_quantumsim").replace("_scheduled", ""))
        except ModuleNotFoundError:
            print(
                "\nThe quantumsim file doesn't exist, so quantumsim cannot be used for simulating this benchmark")
            # raise

    def __exit__(self):
        print("Deleting benchmark garbage")

        delcopy(self.cp)
        delcopy(self.cp.replace(".qasm", "_error.qasm"))

    def __str__(self):

        return "\nQUANTUM BENCHMARK\n"+"\n\tAlgorithm: "+self.qasm_file_path+"\n\tNumber of qubits: "+str(self.N_qubits)+"\n\tNumber of experiment simulations "+str(self.N_exp)

    def error_analysis(self, init_state_type, errprob, quantumsim=False, init_state=""):
        """
        """

        print("############ [Error Analysis] " +
              self.qasm_file_path+" ############")

        N_qubits = self.N_qubits

        if init_state_type == 0:

            for q in range(2**N_qubits):

                with open(INIT_QST_FILE, "w") as f:
                    f.write("0.0 0.0 |"+format(0, "0"+str(N_qubits)+"b")+">\n" +
                            "1.0 0.0 |"+format(q, "0"+str(N_qubits)+"b")[::-1]+">")

                init_state = format(q, "0"+str(N_qubits)+"b")

                # prob_succ, tomography_matrix = analysis(N_qubits, tomography_matrix)
                prob_succ = self.simulate(
                    errprob, quantumsim, init_state)

            print(self.tomography_matrix)

            if N_qubits < 5:
                # It has no sense to see a bar graph or a heatmap of 1024 values or more

                try:
                    just_heatmap(N_qubits, self.tomography_matrix,
                                 self.qasm_file_path.replace(".qasm", ""))
                except MemoryError:
                    print(
                        "Error while drawing the graph. MemoryError despite the matrix size")

        elif init_state_type == 1:

            if init_state == "":

                init_state = "1.0 0.0 |"+format(0, "0"+str(N_qubits)+"b")+">\n"

            with open(INIT_QST_FILE, "w") as f:
                f.write(init_state)
                # norm_factor = 1 / np.sqrt(2**N_qubits)
                # for q in range(2**N_qubits):
                #     f.write(str(norm_factor)+"0.0 |" +
                #             format(q, "0"+str(N_qubits)+"b")[::-1]+">\n")

                self.simulate(errprob, quantumsim=quantumsim,
                              initial_state=init_state)

        else:

            print("\nError. The initial state is missing or is not 0 (all possible inputs), 1 (superosition state) or 2 (special superposition state)")

    def draw_error_analysis(self):
        try:
            graph(self.N_qubits, self.tomography_matrix,
                  self.qasm_file_path.replace(".qasm", ""))
        except MemoryError:
            print("Error while drawing the graph. MemoryError despite the matrix size")

    def probability_of_success(self):

        return sum(self.success_registry)/self.N_exp

    def simulate(self, errprob, quantumsim=False, initial_state=None):

        N_exp = self.N_exp
        qasm_f_path = self.cp

        if quantumsim:          # TODO
            # Quantumsim will be used as simulator

            # print("quantumsim time")

            # expected_q_state, expected_measurement = self.qx_simulation(
            #     qasm_f_path)

            expected_measurement = self.quantumsim_simulation(
                errprob, initial_state)

            # return self.quantumsim_simulation()

            return self.quantumsim_simulation(errprob, initial_state, expected_measurement)

        else:

            expected_q_state, expected_measurement = self.qx_simulation(
                qasm_f_path)

            error_file = qasm_f_path.replace(".qasm", "_error.qasm")
            add_error_model(qasm_f_path, error_file, errprob)

            for i in range(N_exp):

                q_state, measurement = self.qx_simulation(error_file)

                # measurement = measurement[::-1] # for quantumsim maybe?

                # print(expected_q_state)
                # print(q_state)

                print(expected_measurement)
                print(measurement)

                exp_m_int = int(''.join(str(int(e))
                                        for e in expected_measurement.tolist()), 2)
                m_int = int(''.join(str(int(e))
                                    for e in measurement.tolist()), 2)

                self.tomography_matrix[exp_m_int,
                                       m_int] = self.tomography_matrix[exp_m_int, m_int] + 1/N_exp

                self.success_registry.append(1 if np.array_equal(
                    measurement, expected_measurement) else 0)

                self.fidelity_registry.append(
                    self.fidelity(expected_q_state, q_state))

                # Errors while measuring
                meas_errors = np.array(self.fidelity_registry) - \
                    np.array(self.success_registry)
                self.total_meas_err = np.count_nonzero(meas_errors != 0)

            return self.probability_of_success()

    def qx_simulation(self, qasm_f_path):

        qx = qxelarator.QX()

        qx.set(qasm_f_path)

        qx.execute()                            # execute

        # Measure
        c_buff = []
        for q in range(self.N_qubits):
            c_buff.append(qx.get_measurement_outcome(q))

        measurement = np.array(c_buff[::-1], dtype=float)
        print(qx.get_state())
        q_state = self.output_quantum_state(qx.get_state())

        return q_state, measurement

    def quantumsim_simulation(self, error, init_state, expected_measurement=False, meas_error=0.03):

        N_exp = self.N_exp
        N_qubits = self.N_qubits

        if expected_measurement:

            # CIRCUIT DECLARATION
            c = self.qsimc.circuit_function(
                3500, 1500, error, meas_error, init_state)
            # c = self.qsimc.circuit_function(error, meas_error, init_state)

            # SIMULATING
            sdm = sparsedm.SparseDM(c.get_qubit_names())

            measurements = []

            # c.apply_to(sdm)
            # measurements = [sdm.classical["m0"],
            #                 sdm.classical["m1"], sdm.classical["m2"]]

            # return np.array(measurements, dtype=float)

            for i in range(N_exp):
                c.apply_to(sdm)

                # for q in range(N_qubits):
                #     if sdm.classical["m"+str(q)]:
                #         measurements.append(sdm.classical["m"+str(q)])

                for q, m in enumerate(sdm.classical):
                    print("\nMeasurement ID: m"+str(q)+"\n")
                    print(m)
                    measurements.append(m)

                measurement = np.array(measurements, dtype=float)
                print("Expected Measurement:")
                print(expected_measurement)
                print("Actual Measurement:")
                print(measurement)

                exp_m_int = int(''.join(str(int(e))
                                        for e in expected_measurement.tolist()), 2)
                m_int = int(''.join(str(int(e))
                                    for e in measurement.tolist()), 2)
                self.tomography_matrix[exp_m_int,
                                       m_int] = self.tomography_matrix[exp_m_int, m_int] + 1/N_exp

                self.success_registry.append(1 if np.array_equal(
                    measurement, expected_measurement) else 0)

            return self.probability_of_success(self.success_registry, N_exp), self.tomography_matrix

        else:

                                # CIRCUIT DECLARATION
            c = self.qsimc.circuit_function(np.inf, np.inf, 0, 0, init_state)

            # SIMULATING
            sdm = sparsedm.SparseDM(c.get_qubit_names())

            measurements = []

            c.apply_to(sdm)

            # for q in range(N_qubits):
            #     if sdm.classical["m"+str(q)]:
            #         measurements.append(sdm.classical["m"+str(q)])

            for q, m in enumerate(sdm.classical):
                print("\nMeasurement ID: m"+str(q)+"\n")
                print(m)
                measurements.append(m)

            measurement = np.array(measurements, dtype=float)

            return measurement

    def output_quantum_state(self, q_state):
        """ Defines the quantum state based on the output string of QX get_state() function """

        m = re.search(
            r"\(([\+\-]\d[\.\de-]*),([\+\-]\d[\.\de-]*)\) \|(\d+)>", q_state)
        amplitude = complex(float(m.group(1)), float(m.group(2)))

        base_state = np.zeros(2**self.N_qubits)
        base_state[int(m.group(3), 2)] = 1

        return amplitude*base_state

    def check_success(self, expected, actual):

        return 1 if np.array_equal(expected, actual) else 0

    def fidelity(self, expected, actual):
        """ Fidelity calculation """

        f = -1

        if expected.ndim > 1:
            # Super hard calculation.

            print("Expected quantum mixed state detected.")
            print("I'm not ready for this (T.T)")

        elif actual.ndim > 1:
            # Hard calculation

            f = np.sqrt(np.vdot(expected, np.dot(actual, expected)))

        else:
            # Simple calculation

            f = np.absolute(np.vdot(expected, actual))**2

        if f == -1:
            print("Error in the input dimensions. Fidelity not calculated")
            return

        return np.around(f, decimals=5)

    # depth=0 is just for now, till I'm able to extract the depth of the benchmark
    def quantum_volume(self, depth=0):
        """ Quantum Volume calculation"""

        # return min([self.N_qubits, depth])**2
        return self.N_qubits * depth

    def mean_fidelity(self):

        return np.mean(self.fidelity_registry)

    def mean_success(self):

        return np.mean(self.success_registry)
