if __name__ == "__main__":
    from qorange.circuits import QuantumCircuit
    from qorange.gates import *

    x = PauliX()
    y = PauliY()
    z = PauliZ()
    i = Identity()
    swap = SWAP()
    h = Hadamard()
    cnot = CNOT()

    circuit = QuantumCircuit()
    print(circuit.state)
    circuit.apply_gate(1, h)
    print(circuit.state)
    circuit.apply_gate(1, cnot)
    print(circuit.state)


    print("-------------------")

    swap_test = QuantumCircuit()
    swap_test.apply_gate(1, x)
    swap_test.apply_gate(1, swap)
    print(swap_test.state)
