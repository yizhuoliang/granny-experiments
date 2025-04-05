import itertools
import heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional

@dataclass
class TaskSpec:
    id: str
    contention_free_runtime: float  # T_0
    memory_bandwidth: float  # M_0 in MBps
    
    def __repr__(self):
        return f"Task({self.id})"

@dataclass
class RunningTask:
    task: TaskSpec
    work_done_fraction: float

    # NOTE: This is not used across time slices, but only for the current time slice
    # if we implement in a language with pointer, we don't need to have this field
    expected_remaining_time: float = 0.0
    
    def __repr__(self):
        return f"RunningTask({self.task.id}, {self.work_done_fraction*100:.1f}% done)"

def generate_all_possible_schedule(waiting_tasks: List[str]):
    """
    Generate all possible schedules for the waiting tasks.
    """
    return list(set(itertools.permutations(waiting_tasks)))

def simulate_schedule(schedule: List[str], tasks: Dict[str, TaskSpec], num_cores: int, running: List[RunningTask]):
    """
    Simulate a specific given schedule and return the makespan.
    """
    num_total_tasks_to_finish = len(running) + len(schedule)
    num_total_finished_tasks = 0
    time_till_everything_finished = 0.0
    
    while num_total_finished_tasks < num_total_tasks_to_finish:
        assert not (len(running) == 0 and len(schedule) == 0), "No tasks to run and no tasks to schedule"

        # drain all cores
        while len(running) < num_cores and len(schedule) > 0:
            task_id = schedule.pop(0) # start the next task
            running.append(RunningTask(tasks[task_id], 0.0))
        
        # evaluate mem bw pressure at this time slice
        mem_bw_pressure = sum(r.task.memory_bandwidth for r in running)

        # compute the expected finishing times udner the current mem bw pressure
        expected_remaining_times = {}
        for i in range(len(running)):
            r = running[i]
            remaining_contention_free_time = r.task.contention_free_runtime * (1 - r.work_done_fraction)
            expected_remaining_times[i] = slowdown_function(remaining_contention_free_time, mem_bw_pressure)
            r.expected_remaining_time = expected_remaining_times[i]
        
        # find the shortest remaining time
        shortest_remaining_time = min(expected_remaining_times.values())
        
        # tasks might have exactly the same finish time, so we pop all of them
        indices_to_remove = [i for i, time in expected_remaining_times.items() if time == shortest_remaining_time]
        finished_tasks = [running.pop(i) for i in sorted(indices_to_remove, reverse=True)]
        num_total_finished_tasks += len(finished_tasks)

        # update the work done fraction
        for r in running:
            r.work_done_fraction += (1 - r. work_done_fraction) * (shortest_remaining_time / r.expected_remaining_time)
            assert r.work_done_fraction <= 1.0, "Work done fraction cannot exceed 1.0"
        
        # update the time
        time_till_everything_finished += shortest_remaining_time
    
    assert len(running) == 0, "All tasks should have finished"
    return time_till_everything_finished

def find_optimal_schedule(tasks: Dict[str, TaskSpec], num_cores: int, waiting_tasks: List[str], running: List[RunningTask]):

    assert len(running) <= num_cores, "Running tasks cannot exceed the number of cores"
    assert len(waiting_tasks) >= 0, "There must be at least one waiting task"

    shortest_makespan = float("inf")
    optimal_schedule = None
    for schedule in generate_all_possible_schedule(waiting_tasks.copy()):
        makespan = simulate_schedule(list(schedule), tasks, num_cores, running.copy())
        if makespan < shortest_makespan:
            shortest_makespan = makespan
            optimal_schedule = schedule
    return optimal_schedule, shortest_makespan
        

def slowdown_function(t_0, p):
    """
    Model the slowdown due to memory bandwidth contention.
    
    Args:
        t_0: Contention-free runtime
        p: Total memory bandwidth pressure in MBps
        
    Returns:
        Predicted runtime with contention
    """
    return (p / 100) * 0.01 + t_0

def create_test_data():
    task_specs = {
        "f1": TaskSpec(id="f1", contention_free_runtime=10.0, memory_bandwidth=200.0),
        "f2": TaskSpec(id="f2", contention_free_runtime=15.0, memory_bandwidth=150.0),
        "f3": TaskSpec(id="f3", contention_free_runtime=8.0, memory_bandwidth=300.0),
        "f4": TaskSpec(id="f4", contention_free_runtime=12.0, memory_bandwidth=100.0),
        "f5": TaskSpec(id="f5", contention_free_runtime=20.0, memory_bandwidth=50.0),
        "f6": TaskSpec(id="f6", contention_free_runtime=5.0, memory_bandwidth=250.0),
    }

    num_cores = 4
    
    waiting_tasks = ["f2", "f2", "f3", "f5", "f6"]

    running = [
        RunningTask(task_specs["f1"], 0.2),
        RunningTask(task_specs["f4"], 0.5),
    ]

    return task_specs, num_cores, waiting_tasks, running

def main():
    taskSpecs, num_cores, waiting_tasks, running = create_test_data()
    optimal_schedule, makespan = find_optimal_schedule(taskSpecs, num_cores, waiting_tasks, running)
    print(f"Optimal schedule: {optimal_schedule}, Makespan: {makespan}")

if __name__ == "__main__":
    main()