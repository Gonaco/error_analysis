

CREATE TABLE IF NOT EXISTS Benchmarks (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       benchmark VARCHAR(100) NOT NULL,
       source VARCHAR(100) NOT NULL,
       behaviour TEXT,
       N_qubits INT NOT NULL,
       N_gates INT NOT NULL
);

CREATE TABLE IF NOT EXISTS Configurations (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       conf_file VARCHAR(255) NOT NULL,
       scheduler VARCHAR(20) NOT NULL,
       mapper VARCHAR(20) NOT NULL,
       initial_placement VARCHAR(20) NOT NULL
);

CREATE TABLE IF NOT EXISTS HardwareBenchs (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       benchmark INTEGER NOT NULL,
       configuration INTEGER NOT NULL,
       N_qubits INT NOT NULL,
       N_gates INT NOT NULL,
       N_swaps INT NOT NULL,
       depth INT NOT NULL,
       FOREIGN KEY (benchmark)
       REFERENCES Benchmarks (id)
       ON UPDATE RESTRICT
       ON DELETE NO ACTION
       FOREIGN KEY (configuration)
       REFERENCES Configurations (id)
       ON UPDATE NO ACTION
       ON DELETE NO ACTION
);

CREATE TABLE IF NOT EXISTS Results (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       prob_succs DOUBLE NOT NULL,
       mean_f DOUBLE NOT NULL,
       std_f DOUBLE NOT NULL,
       q_vol INT NOT NULL-- ,
       -- map_t DOUBLE NOT NULL
       );

CREATE TABLE IF NOT EXISTS Experiments (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       date DATETIME NOT NULL,
       tom_mtrx_path VARCHAR(255) NOT NULL,
       fail INT DEFAULT 1,
       log_path VARCHAR(255)
       );

CREATE TABLE IF NOT EXISTS SimulationsInfo (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       algorithm INT NOT NULL,       
       simulator VARCHAR(20) NOT NULL,
       N_sim INT NOT NULL,
       error_rate FLOAT,
       t1 INT,
       t2 INT,
       meas_error INT,
       init_type VARCHAR(10) NOT NULL,
       result INT NOT NULL,
       experiment INT NOT NULL,
       FOREIGN KEY (algorithm)
       REFERENCES HardwareBenchs (id)
       ON UPDATE NO ACTION
       ON DELETE NO ACTION
       FOREIGN KEY (result)
       REFERENCES Results (id)
       ON UPDATE NO ACTION
       ON DELETE NO ACTION
       FOREIGN KEY (experiment)
       REFERENCES Experiments (id)
       ON UPDATE RESTRICT
       ON DELETE NO ACTION
);


-- TEST

-- INSERT INTO Experiments (date, tom_mtrx_path)
-- VALUES (datetime('now'),'a.h5');

-- SELECT last_insert_rowid() FROM Experiments;

-- INSERT INTO Results (algorithm, N_sim, init_type, scheduler, error_rate, conf_file, prob_succs, mean_f, exper_id)
-- VALUES ('a.qasm',2,'all_states','ASAP',0.01,'conf.json',0.6,0.6,1);

-- UPDATE Experiments SET fail = 0 WHERE id = 1;

-- SHOWING (SHORT)

-- SELECT benchmark,N_qubits,N_gates,N_sim,init_type,scheduler,error_rate,conf_file,prob_succs,mean_f,date,tom_mtrx_path,fail FROM Results LEFT JOIN Benchmarks ON algorithm = benchmark LEFT JOIN Experiments ON exper_id=Experiments.id;
