import os
from src.modules.jsspProcessor import JSSPProcessor

if __name__ == "__main__":
    dataset_folder = "./src/data/processed/data_20j_15m"

    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            dataset_path = os.path.join(dataset_folder, filename)
            processor = JSSPProcessor(dataset_path)
            best_schedule, best_makespan, exec_time = processor.run()
