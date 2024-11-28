if __name__ == "__main__":
    from qorange.circuits import QuantumCircuit
    from qorange.gates import PauliX

    x = PauliX()

    circuit = QuantumCircuit()
    
    print(circuit.state)
    circuit.apply_gate(1, x)
    print(circuit.state)
