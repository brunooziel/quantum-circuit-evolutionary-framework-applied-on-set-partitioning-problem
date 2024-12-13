import numpy as np
from qiskit.quantum_info import SparsePauliOp
import random
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit.circuit.library import RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate, CRXGate, CRYGate, CRZGate
from qiskit.result import sampled_expectation_value
from qiskit_aer.primitives import Estimator as AerEstimator
import seaborn as sns


#single_gates = [RXGate, RYGate, RZGate] # for the AF-QCE config
single_gates = [RYGate] # for the APCD-QCE config
double_gates = [CRXGate, CRYGate, CRZGate, RXXGate, RYYGate, RZZGate]

def inicial_circuit(qubits):
    
    qc = QuantumCircuit(qubits)
    qubit = np.random.randint(qubits)
    gate = np.random.choice(['rx', 'ry', 'rz'])
    theta = np.random.uniform(0, 2 * np.pi)

    if gate == 'rx':
        qc.rx(theta, qubit)
    elif gate == 'ry':
        qc.ry(theta, qubit)
    elif gate == 'rz':
        qc.rz(theta, qubit)
    
    return qc

def choose_random_gate(circuit):
    
    all_operations = []
    
    for i, instruction in enumerate(circuit.data):
        gate = instruction.operation
        qubits = instruction.qubits
        all_operations.append((i, gate, qubits))
    
    draw = np.random.choice(len(all_operations))
    return all_operations[draw] 

def modify(circuit):
    
    index, gate, qubits = choose_random_gate(circuit)
    epsilon = np.random.normal(0, 0.1)
    new_theta = gate.params[0] + epsilon
    gate = gate.__class__(new_theta)
    circuit.data[index] = (gate, qubits, [])
    
    return circuit, gate.name

def delete(circuit):
    
    index, gate, qubits = choose_random_gate(circuit)
    circuit.data.pop(index)
    
    return circuit, gate.name

def insert(circuit):
    
    if random.choice(['single', 'double']) == 'single':
        
        random_gate_class = random.choice(single_gates)
        angle = random.uniform(-2 * np.pi, 2 * np.pi)  
        random_gate = random_gate_class(angle)
        qubits = [random.choice(range(circuit.num_qubits))]
        
    else:
        
        random_gate_class = random.choice(double_gates)
        angle = random.uniform(-2 * np.pi, 2 * np.pi)  
        random_gate = random_gate_class(angle)
        control_qubit = random.choice(range(circuit.num_qubits))
        target_qubit = random.choice([q for q in range(circuit.num_qubits) if q != control_qubit])
        qubits = [control_qubit, target_qubit]

    depth_positions = list(range(len(circuit.data) + 1))
    insert_position = random.choice(depth_positions)
    
    circuit.data.insert(insert_position, (random_gate, [circuit.qubits[q] for q in qubits], []))
    
    return circuit, random_gate.name

def swap(circuit):
    
    index, gate, qubits = choose_random_gate(circuit)
    circuit.data.pop(index)

    if random.choice(['single', 'double']) == 'single':
        
        random_gate_class = random.choice(single_gates)
        angle = random.uniform(0, 2 * np.pi)  
        random_gate = random_gate_class(angle)
        qubits = [random.choice(range(circuit.num_qubits))]
        
    else:
        
        random_gate_class = random.choice(double_gates)
        angle = random.uniform(0, 2 * np.pi)  
        random_gate = random_gate_class(angle)
        control_qubit = random.choice(range(circuit.num_qubits))
        target_qubit = random.choice([q for q in range(circuit.num_qubits) if q != control_qubit])
        qubits = [control_qubit, target_qubit]

    circuit.data.insert(index, (random_gate, [circuit.qubits[q] for q in qubits], []))
    
    return circuit, random_gate.name

def mutation(circuit):
    
    child = circuit.copy()
    
    actions = ["INSERT", "DELETE", "SWAP", "MODIFY"]
    actions_prob = [0.25, 0.25,0.25, 0.25] # melhor distribuicao que sai do minimo local
    draw = np.random.choice(actions, 1, p=actions_prob)[0]
    gate_type = None

    if len(circuit.data) <= 1:
        
        child, gate_type = insert(child)
        
    else:
        
        if draw == "INSERT":
            child, gate_type = insert(child)
        elif draw == "DELETE":
            child, gate_type = delete(child)
        elif draw == "SWAP":
            child, gate_type = swap(child)
        elif draw == "MODIFY":
            child, gate_type = modify(child)

    return child, (draw, gate_type)

def calculate(estimator, circuit, hamiltonian):

    job = estimator.run(circuits=circuit, observables=hamiltonian)
    results = job.result()
    loss = results.values[0]
        
    return loss

def plots(values, successful_mutations=None, plot_type=True):
    
    if plot_type and successful_mutations is not None:
        # Plot expected values and successful mutations
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

        # Plot expected values over generations
        axes[0].plot(values, '-', color='purple')
        axes[0].set_title('Expected Value of H')
        axes[0].set_xlabel('Generations')
        axes[0].set_ylabel('Energy')
        axes[0].grid(linestyle='--', linewidth=0.5)

        # Process successful mutations data
        mutation_counts = defaultdict(int)
        for mutation, gate in successful_mutations:
            gate_name = gate.replace("Gate", "")
            mutation_counts[(mutation, gate_name)] += 1

        labels, counts = zip(*mutation_counts.items())
        mutations, gates = zip(*labels)

        # Prepare data for Seaborn
        data = pd.DataFrame({
            'Mutation': mutations,
            'Gate': gates,
            'Count': counts
        })

        # Plot successful mutations by gate type
        sns.barplot(data=data, x='Mutation', y='Count', hue='Gate', palette='viridis', ax=axes[1])
        axes[1].set_title("Successful Mutations by Gate Type")
        axes[1].set_ylabel("Successful Count")
        axes[1].set_xlabel("Mutation Type")
        axes[1].grid(axis='x', linestyle='--', linewidth=0.5)
        axes[1].legend()

        # Adjust layout and show plots
        plt.tight_layout()
        plt.show()
        
    else:
        # Plot only the expected values over generations
        plt.plot(values, '-.', color='purple')
        plt.title('Expected Value of H')
        plt.xlabel('Generations')
        plt.ylabel('Energy')
        plt.grid(linestyle='-', linewidth=0.5)
        plt.show()
        
def minimize(estimator, hamiltonian: SparsePauliOp, generations: int, population: int, qc=None, cd_qaoa=None,  ref_value: float = 0, restart=False, plot=False, tol: float = 1e-2):
    
    """Return estimate of energy from estimator
    Parameters:
        estimator: Function to estimate energy
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        generations (int): Total number of generations
        population (int): Number of populations for optimization
        qc: Optional initial quantum circuit
        ref_value (float): Target value of the cost function
        restart (bool): If True, allows resetting in case of stagnation
        plot (bool): If True, plots the results
        tol (float): Tolerance for convergence

    Returns:
        list: Values list of energy estimates
        list: List of generated circuits
    """

    values_list = []
    circuit_list = []
    successful_mutations = []

    current_value = 1e100  # Valor inicial para comparação no primeiro passo

    for step in range(generations):
        
        if step == 0:

            parent = [qc.copy() if qc else inicial_circuit(hamiltonian.num_qubits) for _ in range(population)]

        else:

            parent_c = [current_parent.copy() for _ in range(population)]
            children = [mutation(circuit_) for circuit_ in parent_c]
            
            parent = [result[0] for result in children]
            mutation_info = [result[1] for result in children]
        
        if cd_qaoa:
            cd_parent = [cd_qaoa.compose(cir) for cir in parent]
            
        else:
            cd_parent = parent
            
        loss_list = [calculate(estimator, child, hamiltonian) for child in cd_parent]
        
        if step > 0:
            parent.append(current_parent)
            loss_list.append(current_value)
        
        index = np.argmin(loss_list)

        # Atualiza o circuito atual se um melhor valor for encontrado
        if loss_list[index] <= current_value:
            
            current_parent = parent[index]
            current_value = loss_list[index]
            
            if step > 0 and index < len(parent) - 1:
                successful_mutations.append(mutation_info[index])

        
        # Condição de Restart: se a perda não melhora, procura um valor anterior com diferença > 50
        if restart and step <= generations* 0.6 and len(values_list) >= 250 and len(set(values_list[-250:])) == 1:
            print('\nAttempting restart by locating previous value')
            
            # Procura um valor em values_list com diferença > 50 em relação ao último valor
            found_index = None
            for i in range(len(values_list) - 1, -1, -1):  # Busca reversa
                if (values_list[i] - current_value) > 250:
                    found_index = i
                    break

            # Se um valor válido foi encontrado, reinicia a partir do circuito correspondente
            if found_index is not None:
                print(f"Restarting from previous circuit at step {found_index + 1}")
                current_parent = circuit_list[found_index].copy()
                current_value = values_list[found_index]
    
            else:
                # Caso contrário, reinicia com um circuito inicial
                print('No suitable previous value found, resetting')
                current_parent = inicial_circuit(hamiltonian.num_qubits)
        
                if cd_qaoa:
                    cd_parent = cd_qaoa.compose(current_parent)
                
                else:
                    cd_parent = current_parent
                    
                current_value = calculate(estimator, cd_parent, hamiltonian)
        
        values_list.append(current_value)
        circuit_list.append(current_parent)
            
        print(f"Step: {step + 1} Depth: {current_parent.depth()} Expected Value: {current_value:.7f}", end="\r", flush=True)
        
        if current_value - ref_value <= tol:
            break
        
    if plot: plots(values_list, successful_mutations)

    return values_list, circuit_list
