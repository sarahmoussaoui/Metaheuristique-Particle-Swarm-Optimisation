<!-- @format -->

# Metaheuristique-Particle-Swarm-Optimisation

# Particle Structure in PSO for Job Shop Scheduling (JSSP)

In **Particle Swarm Optimization (PSO)**, each **particle** represents a **potential solution** to the optimization problem, which in the case of **Job Shop Scheduling (JSSP)** is a **schedule** for the jobs and machines.

## 1. Position of a Particle

The **position** of a particle corresponds to a **specific schedule**. In the JSSP, this would mean deciding the **order** in which jobs are assigned to machines.

For example, for a **Job Shop Scheduling Problem (JSSP)**:

- Suppose there are `n` jobs, and each job has to be processed on `m` machines.
- The particle represents a possible **sequence of jobs** on the machines.

### Position Representation:

You could represent the particle’s **position** as a sequence of **job-machine assignments**.
For instance:
Particle Position (Schedule): [Job 1 -> Machine 2, Job 2 -> Machine 1, Job 3 -> Machine 3, ...]

Velocity is a way to guide particles towards better solutions while exploring the solution space.

## 3. Personal Best (pBest)

Each particle keeps track of its **personal best solution** (`pBest`), which is the best schedule it has found during the search process. It compares its current position (schedule) with the `pBest` and updates it if a better solution is found (i.e., a schedule with a smaller makespan).

## 4. Global Best (gBest)

The **global best solution** (`gBest`) is the best schedule found by any particle in the entire swarm. It represents the overall best schedule and guides the search of all particles.

## How the Particle Works in PSO for JSSP:

- At the start of the algorithm, each particle represents a **random schedule**.
- The particle’s **fitness** is evaluated based on how good the schedule is (e.g., using the makespan as the fitness function).
- The **velocity** is updated in each iteration, which influences how the **job-machine assignments** are altered.
- The particle explores different **schedules** by adjusting its **position** (schedule) based on its **velocity** and the guidance from its **personal best** and the **global best**.
- The algorithm iteratively refines the solutions, and over time, the particles converge to an optimal or near-optimal schedule.

## Scheduler in the Particle Structure:

The **scheduler** in this context refers to how the jobs are assigned to machines and in which order. It’s not a traditional scheduler algorithm (like First-Come-First-Serve or Shortest Job Next), but rather, the **position** of each particle encodes a **schedule** (job order and machine assignments), and the particle adjusts these assignments over time using the **PSO algorithm**.

### To break it down:

1. Each **particle** represents a potential schedule (solution).
2. Each **particle’s position** is a specific arrangement of jobs on machines.
3. The **velocity** tells the particle how to modify its schedule.
4. The **personal best** is the best schedule found by that particle so far.
5. The **global best** is the best schedule found by any particle in the swarm.

## Example of a Particle in JSSP:

Let’s assume there are 3 jobs (`Job1`, `Job2`, `Job3`) and 2 machines (`M1`, `M2`).

- A possible **schedule** (particle position) could be:

  - `Job1 -> M1`, `Job2 -> M2`, `Job3 -> M1`

- **Velocity** could represent a swap or change in order, for example:
  - Swap `Job1` with `Job2`, so the new schedule would be:
    - `Job2 -> M1`, `Job1 -> M2`, `Job3 -> M1`

The **fitness** of the particle would be evaluated based on the makespan of this schedule. The goal of the algorithm is to refine the schedule by adjusting the positions of jobs and machines to minimize the makespan.

---

## In summary:

- A **particle** in PSO for JSSP represents a **schedule** for job assignments to machines.
- The **position** of the particle encodes the specific order and machine assignments.
- The **velocity** indicates how the particle’s schedule should change in the next iteration.
- PSO updates the particle's schedule over multiple iterations to search for the best possible solution, balancing **exploration** of new solutions and **exploitation** of the current best ones.

### Valid Schedule Constraints for Job Shop Scheduling Problem (JSSP)

A schedule is **valid** if it follows the constraints of the **Job Shop Scheduling Problem**, which include:

1. **Job Processing Sequence**:

   - A job is processed in a **specific sequence** of machines, based on the order it needs to go through. For example, Job 1 must be processed on Machine 1 first, then on Machine 2, and so on.

2. **Machine Assignment**:
   - A job can only be **assigned to each machine once**. However, a job can be assigned to multiple machines, but it should only be processed on each machine during different time slots (one machine per time slot).
