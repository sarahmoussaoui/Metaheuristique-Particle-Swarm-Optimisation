import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from typing import List, Optional, Union, Tuple


class ScheduleVisualizer:
    """Handles visualization of schedules with improved error handling and type hints."""

    @staticmethod
    def plot_gantt_chart(
        jssp,
        schedule: Optional[List[Tuple[int, int]]] = None,
        figsize: Tuple[Union[int, float], Union[int, float]] = (15.0, 8.0),
        save_folder: Optional[str] = None,
        filename: str = "gantt_chart.png",
        dpi: int = 300
    ):
        """
        Plots a Gantt chart for the JSSP schedule with improved robustness.
        
        Args:
            jssp: JSSP instance containing the problem data
            schedule: Optional sequence of (job_idx, op_idx) tuples to plot
            figsize: Figure size as (width, height) in inches (default: (15.0, 8.0))
            save_folder: Directory to save the plot (None to not save)
            filename: Name for the saved file
            dpi: Resolution for saved image
        """
        try:
            # Convert figsize to ensure proper type
            figsize = (float(figsize[0]), float(figsize[1]))
            
            fig, ax = plt.subplots(figsize=figsize)
            machines = sorted({op.machine for job in jssp.jobs for op in job.operations})
            y_ticks = np.arange(len(machines))
            y_labels = [f"Machine {m}" for m in machines]

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel("Time")
            ax.set_title("Job Shop Schedule Gantt Chart")
            ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.7)

            # Use viridis colormap which is perceptually uniform
            colors = plt.colormaps.get_cmap("viridis", jssp.num_jobs)

            for job in jssp.jobs:
                for op_idx, op in enumerate(job.operations):
                    # Only plot if operation is in schedule or schedule is None
                    if schedule is None or (job.job_id, op_idx) in schedule:
                        if op.start_time is not None and op.end_time is not None:
                            y_pos = machines.index(op.machine)
                            rect = patches.Rectangle(
                                (op.start_time, y_pos - 0.4),
                                op.end_time - op.start_time,
                                0.8,
                                facecolor=colors(job.job_id / jssp.num_jobs),  # Normalized color mapping
                                edgecolor="black",
                                alpha=0.7,
                            )
                            ax.add_patch(rect)
                            ax.text(
                                op.start_time + (op.end_time - op.start_time) / 2,
                                y_pos,
                                f"J{job.job_id + 1}",  # Show 1-based job numbering
                                ha="center",
                                va="center",
                                color="white" if job.job_id / jssp.num_jobs > 0.5 else "black",
                                fontsize=8,
                                fontweight="bold",
                            )

            # Calculate x-axis limit
            end_times = [op.end_time for job in jssp.jobs for op in job.operations if op.end_time is not None]
            max_time = max(end_times) if end_times else 1
            ax.set_xlim(0, max_time * 1.05)

            plt.tight_layout()
            
            if save_folder:
                os.makedirs(save_folder, exist_ok=True)
                save_path = os.path.join(save_folder, filename)
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
                print(f"Gantt chart saved to: {save_path}")
                
        except Exception as e:
            print(f"Error generating Gantt chart: {str(e)}")
        finally:
            plt.close()

    @staticmethod
    def plot_convergence(
        iteration_history: List[int],
        makespan_history: List[float],
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
        figsize: Tuple[Union[int, float], Union[int, float]] = (10.0, 5.0),
        save_folder: Optional[str] = None,
        filename: str = "convergence_plot.png",
        dpi: int = 300
    ):
        """
        Plots the convergence curve with improved styling and error handling.
        
        Args:
            iteration_history: List of iteration numbers
            makespan_history: List of best makespan values
            upper_bound: Optional upper bound reference line
            lower_bound: Optional lower bound reference line
            figsize: Figure size as (width, height) in inches
            save_folder: Directory to save the plot
            filename: Name for the saved file
            dpi: Resolution for saved image
        """
        try:
            # Convert figsize to ensure proper type
            figsize = (float(figsize[0]), float(figsize[1]))
            
            plt.figure(figsize=figsize)
            plt.plot(
                iteration_history,
                makespan_history,
                color="#1f77b4",  # Matplotlib default blue
                linewidth=2,
                label="Best Makespan",
                marker="o",
                markersize=4,
                markevery=5
            )

            # Add reference lines
            if upper_bound is not None:
                plt.axhline(
                    y=upper_bound,
                    color="#d62728",  # Matplotlib default red
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Upper Bound ({upper_bound})",
                )
                
            if lower_bound is not None:
                plt.axhline(
                    y=lower_bound,
                    color="#2ca02c",  # Matplotlib default green
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Lower Bound ({lower_bound})",
                )

            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Makespan", fontsize=12)
            plt.title("Convergence History", fontsize=14, pad=20)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(fontsize=10)
            plt.tight_layout()

            if save_folder:
                os.makedirs(save_folder, exist_ok=True)
                save_path = os.path.join(save_folder, filename)
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
                print(f"Convergence plot saved to: {save_path}")
                
        except Exception as e:
            print(f"Error generating convergence plot: {str(e)}")
        finally:
            plt.close()