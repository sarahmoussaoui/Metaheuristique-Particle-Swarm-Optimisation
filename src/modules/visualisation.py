import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class ScheduleVisualizer:
    """Handles visualization of schedules."""

    @staticmethod
    def plot_gantt_chart(jssp, figsize=(15, 8)):
        """Plots a Gantt chart for the JSSP schedule."""
        fig, ax = plt.subplots(figsize=figsize)
        machines = sorted({op.machine for job in jssp.jobs for op in job.operations})
        y_ticks = np.arange(len(machines))
        y_labels = [f"Machine {m}" for m in machines]

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time")
        ax.set_title("Job Shop Schedule Gantt Chart")
        ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.7)

        colors = plt.cm.get_cmap("tab20", jssp.num_jobs)

        for job in jssp.jobs:
            for op in job.operations:
                if op.start_time is not None and op.end_time is not None:
                    y_pos = machines.index(op.machine)
                    rect = patches.Rectangle(
                        (op.start_time, y_pos - 0.4),
                        op.end_time - op.start_time,
                        0.8,
                        facecolor=colors(job.job_id),
                        edgecolor="black",
                        alpha=0.7,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        op.start_time + (op.end_time - op.start_time) / 2,
                        y_pos,
                        f"J{job.job_id}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

        max_time = max(
            op.end_time
            for job in jssp.jobs
            for op in job.operations
            if op.end_time is not None
        )
        ax.set_xlim(0, max_time * 1.05)

        legend_patches = [
            patches.Patch(color=colors(i), label=f"Job {i}")
            for i in range(jssp.num_jobs)
        ]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_convergence(iteration_history, makespan_history):
        """Plots the makespan improvement over iterations."""
        plt.figure(figsize=(10, 5))
        plt.plot(iteration_history, makespan_history, "b-", linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel("Best Makespan")
        plt.title("Makespan Improvement Over Iterations")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
