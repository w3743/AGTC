"""Compare all-source learned input memory with and without activity retention.

Uses the same paired protocol as memory_scope_benchmark. The historical task
selection is expanded to all nine tasks in the input-memory experiment.
"""
import memory_scope_benchmark as benchmark
from input_memory_benchmark import TASKS


if __name__ == '__main__':
    benchmark.KINDS = ('learned_input_memory20', 'all_memory_fixed_activity20',
                       'all_memory_learned_activity20')
    benchmark.TASKS = tuple(TASKS)
    benchmark.main()
