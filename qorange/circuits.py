import numpy as np
from qorange.gates import Gate, TwoQubitGate, ControlledGate

class QuantumCircuit:
    """
    A simple object-oriented quantum circuit class.

    Attributes:
        num_qubits (int): The number of qubits in the circuit.
        gate_history (list): A list of gates applied to the circuit, stored as tuples (gate_name, target_qubits, parameters).
        state (np.ndarray): A numpy array representing the state vector of the circuit.
    """

    def __init__(self):
        """
        Initializes a quantum circuit with a given number of qubits.

        Args:
            num_qubits (int): The number of qubits in the circuit.
        """
        self.state = np.kron(np.array([1, 0]), np.array([1, 0]))
        self.density_matrix = np.outer(self.state, self.state.conj())
        self.gate_history = []  # Stores the gates applied to the circuit

    def update_state(self, new_state):
        """
        Updates the state of the circuit.

        Args:
            new_state (np.ndarray): The new state of the circuit.
        """
        self.state = new_state
        self.density_matrix = np.outer(self.state, self.state.conj())

    def apply_gate(self, q_index, gate):
        """
        Applies a gate matrix to the circuit's state vector.

        Args:
            q_index: int if single qubit operation (1 for q1 and 2 for q2)
            gate: gate being applied
        """
        if isinstance(gate, Gate):
            gate_info = { "gate": gate, "target": q_index }
            if isinstance(gate, TwoQubitGate):
                # q_index is not necessary here.
                gate_matrix = gate.matrix
                gate_info["control"] = 1 if q_index == 2 else 2
            else:
                if q_index == 1:
                    gate_matrix = np.kron(gate.matrix, np.eye(2))
                    gate_info["control"] = None
                elif q_index == 2:
                    gate_matrix = np.kron(np.eye(2),gate.matrix)
                    gate_info["control"] = None
                else:
                    raise Exception("Invalid indexing of qubits")

            self.update_state(np.matmul(gate_matrix, self.state))
            self.gate_history.append(gate_info)

        elif isinstance(gate, ControlledGate):
            if q_index == 1:
                # control is on the first qubit
                gate_matrix = np.kron(np.array([[1,0],[0,0]]), np.eye(2)) + np.kron(np.array([[0,0],[0,1]]), gate.get_matrix())
            elif q_index == 2:
                # control is on the second qubit
                gate_matrix = np.kron(np.eye(2), np.array([[1,0],[0,0]])) + np.kron(gate.get_matrix(), np.array([[0,0],[0,1]]))
            else:
                raise Exception("Invalid indexing of qubits")
            
            self.update_state(np.matmul(gate_matrix, self.state))
            self.gate_history.append({
                "gate": gate,
                "target": 1 if q_index == 2 else 2,
                "control": q_index,
            })

        else:
            raise Exception("Specified gate is invalid, use Gate or ControlledGate class")

    def measure(self):
        """
        Measures the state of the circuit.

        Returns:
            list: A list representing the probabilities of each basis state.
        """
        probabilities = np.abs(self.state) ** 2
        return probabilities.tolist()

    def draw(self):
        """
        Draws a circuit diagram using ASCII characters.
        """
        EMPTY_SEGMENT = [
            "           ",
            "───────────",
            "           ",
        ]

        diagram = [
            [], # Qubit 1 line
            [], # Qubit 2 line
        ]

        for gate_info in self.gate_history:
            control_qubit = gate_info["control"]
            target_qubit = gate_info["target"]

            if control_qubit:
                num_gates_control = len(diagram[control_qubit - 1])
                num_gates_target = len(diagram[target_qubit - 1])
                if (num_gates_control > num_gates_target):
                    # Add some padding to the target line to make sure everything aligns
                    diagram[target_qubit - 1].extend(
                        [EMPTY_SEGMENT for i in range(num_gates_control - num_gates_target)]
                    )
                elif (num_gates_control < num_gates_target): 
                    # Add some padding to the control line to make sure everything aligns
                    diagram[control_qubit - 1].extend(
                        [EMPTY_SEGMENT for i in range(num_gates_target - num_gates_control)]
                    )
                diagram[control_qubit - 1].append(
                    gate_info["gate"].draw(control_qubit, is_target=False)
                )

            diagram[target_qubit - 1].append(
                gate_info["gate"].draw(target_qubit, is_target=True)
            )

        if len(diagram[0]) < len(diagram[1]):
            # Pad the end of qubit line 1 to match the length of qubit line 2
            diagram[0].extend([EMPTY_SEGMENT for i in range(len(diagram[1]) - len(diagram[0]))])
        elif len(diagram[0]) > len(diagram[1]):
            # Pad the end of qubit line 2 to match the length of qubit line 1
            diagram[1].extend([EMPTY_SEGMENT for i in range(len(diagram[0]) - len(diagram[1]))])

        for qubit_line_n in range(2):
            for printed_line_m in range(3):
                line = "".join([
                    printed_lines[printed_line_m] for printed_lines in diagram[qubit_line_n]
                ])
                print(line)
