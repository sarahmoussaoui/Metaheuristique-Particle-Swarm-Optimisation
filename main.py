import os
from src.modules.jsspProcessor import JSSPProcessor

if __name__ == "__main__":
    dataset_folder = "./src/data/processed/data_20j_15m"

    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            dataset_path = os.path.join(dataset_folder, filename)
            processor = JSSPProcessor(dataset_path)
            best_schedule, best_makespan, exec_time = processor.run(
                num_particles=100,
                max_iter=300,
                w=0.7,
                c1=1.5,
                c2=1.5,
                adaptive_params=True,
                mutation_rate=0.2,
                max_stagnation=40,
                early_stopping_window=40,
                improvement_threshold=0.01,
            )
