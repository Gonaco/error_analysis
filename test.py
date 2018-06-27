import quantum_benchmark
import pickle

with open('benchmark_objects_saving', 'wb') as output:

    # b = quantum_benchmark.Benchmark("3_17_13.qasm")  # error
    # b = quantum_benchmark.Benchmark("ham3_102.qasm")
    # b = quantum_benchmark.Benchmark("vbeAdder_1b.qasm")
    b = quantum_benchmark.Benchmark("alu-bdd_288.qasm",2)
    
    print(b)
    # input()
    
    b.error_analysis(0,0.01)
    print("\n---Benchmark characteristics")
    print(b.qasm_file_path)
    print(b.success_registry)
    print(b.fidelity_registry)
    print(b.total_meas_err)
        
    # pickle.dump(b, output, pickle.HIGHEST_PROTOCOL)
    
    # b.error_analysis(1,0.01,False,"0.57 0.0 |000>\n0.57 0.0 |111>\n")
    # print("\n---Benchmark characteristics")
    # print(b.qasm_file_path)
    # print(b.success_registry)
    # print(b.fidelity_registry)
    # print(b.total_meas_err)
