import os
from src.modules.jsspProcessor import JSSPProcessor

if __name__ == "__main__":
    dataset_folder = "./src/data/processed/data_50j_15m"

    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            dataset_path = os.path.join(dataset_folder, filename)
            processor = JSSPProcessor(dataset_path)
            best_schedule, best_makespan, exec_time = processor.run(
                num_particles=175,
                max_iter=1000,
                w=0.5,
                c1=0.4,
                c2=0.9,
                adaptive_params=True,
                mutation_rate=0.7,
                max_stagnation=20,
                early_stopping_window=None,
                improvement_threshold=0.08,
            )
