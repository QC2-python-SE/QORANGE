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
    circuit.apply_gate(cnot, (1,2))
    
    circuit.apply_gate(cnot, (2,1))
    circuit.apply_gate(CZ(), (2,1))
    circuit.draw()
    
