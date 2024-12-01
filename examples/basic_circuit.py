if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('..', 'qorange')))
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

    circuit.apply_gate(h, 1)
    circuit.apply_gate(swap)
    circuit.draw()
    
