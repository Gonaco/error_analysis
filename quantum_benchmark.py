import numpy as np
import os
import atexit
import re
import importlib.util


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

# DATABASE ####################################################################

import h5py
from datetime import datetime
import sqlite3
connection = sqlite3.connect("error_analysis.db")

# CONSTANTS ###################################################################

PURE_OPT = 0
SCHED_OPT = 1
MAPP_OPT = 2

INIT_QST_FILE = "init_state.qst"

ALL_STAT = 0
ONE_STAT = 1

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

class MappingAnalysis(object):

    '''Object that runs the error analysis of some benchmarks'''

    def __init__(self, benchmarks, db_path, log_path, h5_path, output_dir_name, init_type, error, simulator, t1, t2, meas_error):

        self.connection = sqlite3.connect(db_path)
        self.benchmarks = benchmarks
        self.log_path = log_path
        self.h5_path = h5_path
        self.cursor = self.connection.cursor()
        self.init_type = init_type
        self.error = error
        self.simulator = simulator

    def save_in_db(self, benchmark, source, prob_succs, mean_f, q_vol):

        if self.init_type == 0:
            init_type = "all_states"
        elif self.init_type == 1:
            init_type = "ground_state"
        else:
            print("Init type ERROR. The init_type is not either 1 nor 0")
            raise Exception("Init_TypeError")

        if self.simulator:
            simulator = "quantumsim"
        else:
            simulator = "qx"

        format_str = "INSERT INTO Benchmarks (benchmark, source, N_swaps, depth, N_sim, init_type, scheduler, mapper, initial_placement, error_rate, conf_file) VALUES ('{name}', '{source}', {N_swaps}, {depth}, {N_sim}, '{init_type}', '{scheduler}', '{mapper}', '{initial_placement}', {error_rate}, '{conf_file}');"

        name, N_swaps, depth, N_sim, scheduler, mapper, initial_placement, conf_file = benchmark.getAll()

        self.cursor.execute(format_str.format(name=name, N_swaps=N_swaps, depth=depth, N_sim=N_sim, init_type=self.init_type,
                                              scheduler=scheduler, mapper=mapper, initial_placement=initial_placement, error_rate=self.error_rate, conf_file=conf_file))
        self.connection.commit()
        self.cursor.execute("SELECT last_insert_rowid() FROM Experiments;")
        algorithm = cursor.fetchone()[0]

        format_str = "INSERT INTO Results (algorithm, prob_succs, mean_f, q_vol, simulator, exper_id) VALUES ({prob_succs}, {mean_f}, {q_vol}, {simulator}, {exper_id});"

        self.cursor.execute(format_str.format(algorithm=algorithm, prob_succs=prob_succs,
                                              mean_f=mean_f, q_vol=q_vol, exper_id=self.exper_id, simulator=self.simulator))

    def db_init_query(self):

        new_experiment_query = "INSERT INTO Experiments (date, tom_mtrx_path, log_path) VALUES (datetime('now'),'" +
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

    def db_genesis(self):
        '''Check wether the database exits or not and if it does not exist it creates it'''

    def analysis(self):

        with h5py.File(self.h5_path, "w") as h5f:

            self experiment_id = self.db_init_query()

            try:

                for benchmark in self.benchmarks:

                    if self.simulator:  # Quantumsim
                        benchmark.quantumsim_mapped.error_analysis(
                            self.init_type, error, init_state, t1, t2, meas_error)
                    else:       # QX
                        benchmark.cqasmm_mapped.error_analysis(
                            self.init_type, error, init_state)

                    # self.descripBench = _DescripBench(alg_path)
                    # self.descripBench.compile(config_file_path)
                    # self.simBench = _SimBench(qasm_file_path)

            except:
                self.db_interruption_query()
                raise

        self.db_final_query()


class Benchmark(object):
    '''The Benchmark class describes the benchmark and contains all its desciptions (OpenQL, cQASM and quantumsim)'''

    def __init__(self, openql_file_path, source, config_file_path, scheduler="ALAP", mapper="minextendrc", initial_placement="no", output_dir_name="benchmarks_exports", N_exp=1000):

        self.name = os.path.split(openql_file_path)[1]
        self.source = source
        self.N_exp = N_exp
        self.ql_descr = _DescripBench(
            openql_file_path, config_file_path, scheduler, mapper, initial_placement, output_dir_name)

        self.cqasm_pure, self.cqasm_sched, self.cqasm_mapped, self.quantumsim_sched, self.quantumsim_mapped = self.ql_descr.compile(
            N_exp)

    def getConfig(self):
        return self.ql_descr.config_file_path

    def getScheduler(self):
        return self.ql_descr.scheduler

    def getMapper(self):
        return self.ql_descr.mapper

    def getInitPlace(self):
        return self.ql_descr.init_place

    def getOutputDir(self):
        return self.ql_descr.output_dir

    def getSimBench(self):
        return [self.cqasm_pure, self.cqasm_sched, self.cqasm_mapped, self.quantumsim_sched, self.quantumsim_mapped]

    def getN_qubits(self, option=-1):
        if option == PURE_OPT:
            return self.cqasm_pure.N_qubits
        elif option == SCHED_OPT:
            return self.cqasm_sched.N_qubits
        elif option == MAPP_OPT:
            return self.cqasm_mapped.N_qubits
        else:
            return [self.cqasm_pure.N_qubits, self.cqasm_sched.N_qubits, self.cqasm_mapped.N_qubits]

    def getN_gates(self, option=-1):
        if option == PURE_OPT:
            return self.cqasm_pure.N_gates
        elif option == SCHED_OPT:
            return self.cqasm_sched.N_gates
        elif option == MAPP_OPT:
            return self.cqasm_mapped.N_gates
        else:
            return [self.cqasm_pure.N_gates, self.cqasm_sched.N_gates, self.cqasm_mapped.N_gates]

    def getN_swaps(self, option=-1):
        if option == PURE_OPT:
            return self.cqasm_pure.N_swaps
        elif option == SCHED_OPT:
            return self.cqasm_sched.N_swaps
        elif option == MAPP_OPT:
            return self.cqasm_mapped.N_swaps
        else:
            return [self.cqasm_pure.N_swaps, self.cqasm_sched.N_swaps, self.cqasm_mapped.N_swaps]

    def getDepth(self, option=-1):
        if option == PURE_OPT:
            return self.cqasm_pure.depth
        elif option == SCHED_OPT:
            return self.cqasm_sched.depth
        elif option == MAPP_OPT:
            return self.cqasm_mapped.depth
        else:
            return [self.cqasm_pure.depth, self.cqasm_sched.depth, self.cqasm_mapped.depth]

    def getAll(self):

        return self.name, self.source, self.cqasm_mapped.N_swaps, self.cqasm_mapped.depth self.N_exp, self.ql_descr.scheduler, self.ql_descr.mapper, self.ql_descr.init_place, self.ql_descr.config_file_path


class _QASMReader(object):
    '''An object able to read and store information from a QASM file'''

    def __init__(self, file_path):

        self.file_path = file_path
        with open(file_path, "r") as i:
            self.data = i.readlines()

        for idx, line in enumerate(self.data):

            self.extractInfo(line)

    def save(self, output_path):

        with open(output_path, "w") as o:
            o.writelines(self.data)

    def search(self, regex, line):

        return re.search(regex, line)

    def extractInfo(self, line):

        # Searhc for the info and store it
        self.searchN_qubits(line)
        self.searchDepth(line)
        self.searchN_gates(line)
        self.searchN_swaps(line)

    def searchDepth(self, line):

        a = self.search("# Total depth: (\d+)", line)
        b = self.search("depth: (\d+)", line)

        if a:
            self.depth = int(a[1])
        elif b:
            self.depth = int(b[1])

    def searchN_qubits(self, line):

        a = self.search("^# Qubits used: (\d*)", line)
        b = self.search("^qubits (\d*)", line)
        c = self.search("qubits used: (\d*)", line)

        if a:
            self.N_qubits = int(a[1])
        elif b:
            self.N_qubits = int(b[1])
        elif c:
            self.N_qubits = int(c[1])

    def searchN_gates(self, line):

        a = self.search(
            "# Total no. of quantum gates: (\d+)", line)
        b = self.search(
            "quantum gates: (\d+)", line)

        if a:
            self.N_gates = int(a[1])
        elif b:
            self.N_gates = int(b[1])

    def searchN_swaps(self, line):

        a = self.search(
            "swaps added: (\d*)", line)

        if a:

            self.N_swaps = int(a[1])

    def addinit(self):
        """
        """

        init = '\n.init\n    load_state "'+INIT_QST_FILE+'"\n'
        self.add2qasm("^qubits \d+", init)

        # self.add_measurement()

    def add2qasm(self, before, after):
        """Look for some regular expression in a file (before)
        and add something new after it in a copy of this file
        """

        for idx, line in enumerate(self.data):

            if re.search(before, line):
                self.data[idx] = line+"\n"+after

    def add_error_model(self, errprob):

        error_model = "error_model depolarizing_channel, " + str(errprob)
        self.add2qasm("^qubits \d+", error_model)

    def isQasm(self):

        if "qasm" in self.file_path:

            return True

    # ------------ OLD STAFF

    def check_cQasm(self, filename):

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

    def num_qubitsChecker(self, line, num_qubits):

        correction = line

        if "qubits" in line:
            correction = "qubits {num_qubits}\n".format(num_qubits=num_qubits)

            return correction
        else:

            return False

    def bigQubitNum(self, line, biggest_number):

        match = re.findall(r"q\[?(\d+)\]?", line)

        # print("\nmatch:")
        # print(match)

        # print("\nmatch2int:")
        # print(list(map(int, match)))

        if match:
            '''The biggest qubit number between the biggest of all the qubit numbers in a line and the previous biggest number'''
            biggest_number = max(max(list(map(int, match))), biggest_number)

        return biggest_number

    def algNameChecker(self, line):

        correction = doubleNameChecker(line)

        return numberNameChecker(correction)

    def doubleNameChecker(self, line):

        correction = line
        before = r"(\..+)\..+$"
        after = r"\1"

        c = re.sub(before, after, correction)

        if c:
            correction = c

        return correction

    def numberNameChecker(self, line):

        correction = line
        before = r"^\.(\d.+)"
        after = r".kernel_\1"

        c = re.sub(before, after, correction)

        if c:
            correction = c

        return correction

    def versionChecker(self, corrected):

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

    def parenthesisChecker(self, line):

        correction = line
        before = r"q(\d+)"
        after = r"q[\1]"

        c = re.sub(before, after, correction)

        if c:
            correction = c

        return correction

    def rotationGatesChecker(self, line):

        correction = line
        # There could be two different kind of rotations, + and - 90
        before = [r"r([xy])(\d+) (q\[?\d+\]?)", r"([xy])m(\d+) (q\[?\d+\]?)"]
        after = [r"r\1 \3, \2", r"r\1 \3, -\2"]

        for i, b in enumerate(before):

            c = re.sub(b, after[i], correction)

            if c:
                correction = c

        return correction

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


class _DescripBench(object):

    '''Object for taking care of the OpenQL benchmarks'''

    def __init__(self, file_path, config_file_path, scheduler, mapper, initial_placement, output_dir_name):

        self.file_path = file_path
        self.config_file_path = config_file_path
        self.scheduler = scheduler
        self.mapper = mapper
        self.init_place = initial_placement
        self.output_dir = output_dir_name

        self.openql = importlib.util.spec_from_file_location(
            self.file_path.replace(".py", ""), self.file_path)
        self.openql_comp = importlib.util.module_from_spec(self.openql)

    def compile(self, N_exp):

        uniform_sched = "no"
        scheduler = self.scheduler

        self.openql.loader.exec_module(self.openql_comp)

        if "uniform" in self.scheduler:
            uniform_sched = "yes"
            scheduler = "ASAP"

        try:
            self.openql_comp.circuit(
                self.config_file_path, scheduler, uniform_sched, self.mapper, self.init_place, self.output_dir)

            return _SimBench(self.file_path.replace(".py", ".qasm"), N_exp, self.output_dir), _SimBench(self.file_path.replace(".py", "_scheduled.qasm"), N_exp, self.output_dir), _SimBench(self.file_path.replace(".py", "_rcscheduler_out.qasm"), N_exp, self.output_dir), _SimBench(self.file_path.replace(".py", "_quantumsim_.py"), N_exp, self.output_dir), _SimBench(self.file_path.replace(".py", "_quantumsim_mapped.py"), N_exp, self.output_dir)

        except TypeError:
            print("\nERROR. Configuration file has not all the required definitions." +
                  "\nIn the case of a non-defined gate, the output will be compiled and the result will have that gate")
            raise


class _SimBench(object):

    '''Class for simulating the Benchmark'''

    def __init__(self, file_path, N_exp=1000, out_dir=""):

        file_path = os.path.split(file_path)
        self.file_path = os.path.join(file_path[0], out_dir, file_path[1])
        self.cp = os.path.join(file_path[0], out_dir, "."+file_path[1]+"~")

        self.reader = _QASMReader(self.file_path)

        try:
            self.N_qubits = self.reader.N_qubits
            self.N_gates = self.reader.N_gates
            self.N_swaps = self.reader.N_swaps
            self.depth = self.reader.depth
        except AttributeError:
            self.N_swaps = -1

        self.N_exp = N_exp
        self.success_registry = []  # Matrix storing the success
        self.fidelity_registry = []  # Matrix storing the fidelity
        self.total_meas_err = 0

        if self.reader.isQasm():
            # Initializing qasm copy

            try:
                self.reader.addinit()
                self.reader.save(self.cp)

            except FileNotFoundError:
                print(
                    "\nThe QASM file does not exist or the path is incorrect." +
                    "\nThe Benchmark cannot be created\n")
                raise

            self.quantusim = False

        else:
            # Initializing quantumsim

            try:
                qsim = importlib.util.spec_from_file_location(
                    file_path[1].replace(".py", ""), self.file_path)
                self.qsimc = importlib.util.module_from_spec(qsim)
                qsim.loader.exec_module(self.qsimc)

                # self.qsimc = __import__(
                #     self.file_path.replace(".py", ""))

            except ModuleNotFoundError:
                print(
                    "\nThe quantumsim file doesn't exist, so quantumsim cannot be used for simulating this benchmark")
                raise

            self.quantusim = True

        # atexit.register(delcopy, cp="."+qasm_file_path+"~")
        # atexit.register(delcopy, cp="."+qasm_file_path.replace(
        #     ".qasm", "_error.qasm")+"~")

        self.tomography_matrix = np.zeros((2**self.N_qubits, 2**self.N_qubits))

    def __exit__(self):
        print("Deleting benchmark garbage")

        delcopy(self.cp)
        delcopy(self.cp.replace(".qasm", "_error.qasm"))

    def __str__(self):

        return "\nQUANTUM BENCHMARK\n"+"\n\tAlgorithm: "+self.file_path+"\n\tNumber of qubits: "+str(self.N_qubits)+"\n\tNumber of experiment simulations "+str(self.N_exp)

    def error_analysis(self, init_state_type, errprob, init_state="", t1=3500, t2=1500, meas_error=0.03):
        """
        """

        print("############ [Error Analysis] " +
              self.file_path+" ############")

        N_qubits = self.N_qubits

        if init_state_type == 0:

            for q in range(2**N_qubits):

                with open(INIT_QST_FILE, "w") as f:
                    f.write("0.0 0.0 |"+format(0, "0"+str(N_qubits)+"b")+">\n" +
                            "1.0 0.0 |"+format(q, "0"+str(N_qubits)+"b")[::-1]+">")

                init_state = format(q, "0"+str(N_qubits)+"b")

                prob_succ = self.simulate(
                    errprob, self.quantumsim, init_state, t1, t2, meas_error)

            print(self.tomography_matrix)

            if N_qubits < 5:
                # It has no sense to see a bar graph or a heatmap of 1024 values or more

                try:
                    just_heatmap(N_qubits, self.tomography_matrix,
                                 self.file_path.replace(".qasm", ""))
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

                # self.simulate(errprob, quantumsim=quantumsim,
                #               initial_state=init_state)

                self.simulate(errprob,
                              initial_state=init_state, t1=t1, t2=t2, meas_error=meas_error)

        else:

            print("\nError. The initial state is missing or is not 0 (all possible inputs), 1 (superosition state) or 2 (special superposition state)")

    def draw_error_analysis(self):
        try:
            graph(self.N_qubits, self.tomography_matrix,
                  self.file_path.replace(".qasm", ""))
        except MemoryError:
            print("\nError while drawing the graph. MemoryError despite the matrix size")

    def probability_of_success(self):

        return sum(self.success_registry)/self.N_exp

    def simulate(self, errprob, initial_state=None, t1=3500, t2=1500, meas_error=0.03):

        N_exp = self.N_exp
        qasm_f_path = self.cp

        if self.quantumsim:          # TODO
            # Quantumsim will be used as simulator

            expected_measurement, expected_q_state = self.quantumsim_simulation(
                errprob, initial_state)

            return self.quantumsim_simulation(errprob, initial_state, expected_measurement, expected_q_state, t1, t2, meas_error)

        else:

            expected_q_state, expected_measurement = self.qx_simulation(
                qasm_f_path)

            error_file = qasm_f_path.replace(".qasm", "_error.qasm")
            add_error_model(qasm_f_path, error_file, errprob)

            for i in range(N_exp):

                q_state, measurement = self.qx_simulation(error_file)

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

    def quantumsim_simulation(self, error, init_state, expected_measurement=np.array([]), expected_q_state=0, meas_error=0.03):

        N_exp = self.N_exp
        N_qubits = self.N_qubits

        if expected_measurement.size == 0:

            # CIRCUIT DECLARATION
            c = self.qsimc.circuit_function(np.inf, np.inf, 0, 0, init_state)

            # SIMULATING
            sdm = sparsedm.SparseDM(c.get_qubit_names())

            measurements = []

            c.apply_to(sdm)

            # for q in range(N_qubits):
            #     if sdm.classical["m"+str(q)]:
            #         measurements.append(sdm.classical["m"+str(q)])

            for q in sdm.classical:
                if "m" in str(q):
                    measurements.append(sdm.classical[str(q)])

            measurement = np.array(measurements, dtype=float)
            expected_q_state = sdm.full_dm.dm.ravel()

            return measurement, expected_q_state

        else:

                        # CIRCUIT DECLARATION
            c = self.qsimc.circuit_function(
                3500, 1500, error, meas_error, init_state)
            # c = self.qsimc.circuit_function(error, meas_error, init_state)

            for i in range(N_exp):

                # SIMULATING
                sdm = sparsedm.SparseDM(c.get_qubit_names())
                c.apply_to(sdm)

                measurements = []

                # for q in range(N_qubits):
                #     if sdm.classical["m"+str(q)]:
                #         measurements.append(sdm.classical["m"+str(q)])

                for q in sdm.classical:
                    if "m" in str(q):
                        measurements.append(sdm.classical[str(q)])

                measurement = np.array(measurements, dtype=float)
                print("Expected Measurement:")
                print(expected_measurement)
                print("Actual Measurement:")
                print(measurement)

                q_state = sdm.full_dm.dm.ravel()

                exp_m_int = int(''.join(str(int(e))
                                        for e in expected_measurement.tolist()), 2)
                m_int = int(''.join(str(int(e))
                                    for e in measurement.tolist()), 2)
                self.tomography_matrix[exp_m_int,
                                       m_int] = self.tomography_matrix[exp_m_int, m_int] + 1/N_exp

                self.fidelity_registry.append(
                    np.dot(expected_q_state, q_state))

                self.success_registry.append(1 if np.array_equal(
                    measurement, expected_measurement) else 0)

            # return self.probability_of_success(self.success_registry, N_exp), self.tomography_matrix
            return self.probability_of_success(), self.tomography_matrix

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
    def q_vol(self):
        """ Quantum Volume calculation"""

        # return min([self.N_qubits, depth])**2
        return self.N_qubits * self.depth

    def mean_fidelity(self):

        return np.mean(self.fidelity_registry) if self.fidelity_registry else -1

    def mean_success(self):

        return np.mean(self.success_registry)
