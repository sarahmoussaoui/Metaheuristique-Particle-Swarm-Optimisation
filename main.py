import os
from src.modules.jsspProcessor import JSSPProcessor

if __name__ == "__main__":

    dataset_path = "./src/data/processed/data_20j_15m/data_20j_15m_1.txt"  # Replace with your dataset path
    processor = JSSPProcessor(dataset_path)
    best_schedule, best_makespan, exec_time = processor.run(
        num_particles=10, max_iter=50, w=0.7, c1=2.0, c2=1.0, use_spt=True
     )
