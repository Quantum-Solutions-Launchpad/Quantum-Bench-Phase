from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from _real_space_simulated_common import main


if __name__ == "__main__":
    main("noisy", backend=FakeSherbrooke())
