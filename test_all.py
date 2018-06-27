import importlib.util
import os
import quantum_benchmark

directory_path = ""
filename = "ham3_102.py"
config_file_path = ""
scheduler = "ASAP"
output_dir_name = "qasm_out"

init_state_type = 0             # All states
err_prob = 0.01

# ----------------------------------------------------------------------

openql = importlib.util.spec_from_file_location(filename.replace(
    ".py", ""), os.path.join(os.path.dirname(directory_path), filename))
openql_comp = importlib.util.module_from_spec(openql)
openql.loader.exec_module(openql_comp)
try:
    openql_comp.circuit(
            config_file_path, scheduler, output_dir_name)
except TypeError:
    print("\nERROR. Configuration file has not all the required definitions." +
                      "\nIn the case of a non-defined gate, the output will be compiled and the result will have that gate")
    raise
except AttributeError:
    print("\nERROR. "+filename+" has no circuit to compile")
    raise

b = quantum_benchmark.Benchmark(os.path.joint(os.path.dirname(output_dir_name), filename))
b.error_analysis(init_state_type, err_prob)
