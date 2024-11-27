import numpy as np
from gateClass import Gate, ControlledGate, SWAP

class QuantumCircuit:
    """
    A simple object-oriented quantum circuit class.

    Attributes:
        num_qubits (int): The number of qubits in the circuit.
        gates (list): A list of gates applied to the circuit, stored as tuples (gate_name, target_qubits, parameters).
        state (np.ndarray): A numpy array representing the state vector of the circuit.
    """

    def __init__(self):
        """
        Initializes a quantum circuit with a given number of qubits.

        Args:
            num_qubits (int): The number of qubits in the circuit.
        """

        #self.q1 = np.array([1, 0])
        #self.q2 = np.array([1, 0])
        self.state = np.kron(np.array([1, 0]), np.array([1, 0]))
        self._gates = []  # Stores the gates applied to the circuit
        # self.state = np.zeros((2 ** num_qubits,), dtype=complex)
        # self.state[0] = 1

        

   
       

    def apply_gate(self, gate, q_index):
        """
        Applies a gate matrix to the circuit's state vector.

        Args:
            q_index: int if single qubit operation (1 for q1 and 2 for q2)
            gate: gate being applied
        """
        if isinstance(gate, Gate):
            if isinstance(gate, TwoQubitGate):
                # q_index is not necessary here.
                gate_matrix = gate.matrix
            else:
                if q_index == 1:
                    gate_matrix = np.kron(gate.matrix, np.eye(2))
                elif q_index == 2:
                    gate_matrix = np.kron(np.eye(2),gate.matrix)
                else:
                    raise Exception("Invalid indexing of qubits")
            self.state = np.matmul(gate_matrix, self.state)
        elif isinstance(gate, ControlledGate):
            if q_index == 1:
                # control is on the first qubit
                gate_matrix = np.kron([[1,0],[0,0]], gate.get_matrix()) + np.kron([[0,0],[0,1]], gate.get_matrix())
            elif q_index == 2:
                # control is on the second qubit
                gate_matrix = np.kron(gate.get_matrix(), [[1,0],[0,0]]) + np.kron(gate.get_matrix(), [[0,0],[0,1]])
            else:
                raise Exception("Invalid indexing of qubits")
            
            self.state = np.matmul(gate_matrix, self.state)
        else:
            raise Exception("Specified gate is invalid, use Gate or ControlledGate class")
        
        
        
    
        
    def x(self, target):
        """
        Applies an X (NOT) gate to a qubit.

        Args:
            target (int): The qubit to apply the X gate on.
        """
        x_matrix = np.array([[0, 1],
                             [1, 0]])
        self.apply_gate(x_matrix, target)

    def h(self, target):
        """
        Applies a Hadamard (H) gate to a qubit.

        Args:
            target (int): The qubit to apply the H gate on.
        """
        h_matrix = np.array([[1,  1],
                             [1, -1]]) / np.sqrt(2)
        self.apply_gate(h_matrix, target)

    def measure(self):
        """
        Measures the state of the circuit.

        Returns:
            list: A list representing the probabilities of each basis state.
        """
        probabilities = np.abs(self.state) ** 2
        return probabilities.tolist()

   

if __name__ == "__main__":
    from gateClass import *

    x = PauliX()
    y = PauliY()
    z = PauliZ()
    i = Identity()
    cnot = CNOT()

    circuit = QuantumCircuit()
    circuit.apply_gate(1, x)
    