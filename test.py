from quantum_benchmark import Benchmark
from quantum_benchmark import MappingAnalysis

a = Benchmark("/home/dmorenomanzano/qbench/mapping_benchmarks/4gt11_82.py",
              "/home/dmorenomanzano/qbench/config_files/constraints_configuration_quantumsim_sc17.json")
test = MappingAnalysis([a], "test.db", 1)
test.analyse(True, 10, 0.1, 3500, 1500, 0.3)
