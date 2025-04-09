import os
from src.modules.jsspProcessor import JSSPProcessor

if __name__ == "__main__":
    dataset_path = "./src/data/processed/data_20j_15m/data_20j_15m_1.txt"  
    
    processor = JSSPProcessor(dataset_path)
    
    best_schedule, best_makespan, exec_time = processor.run(
        num_particles=50, 
        max_iter=500, 
        w=0.7, 
        c1=2.0, 
        c2=1.0, 
        use_spt=True,
        spt_mutation_rate=0.4  
    )
    
    print(f"Meilleur makespan obtenu : {best_makespan}")
    print(f"Temps d'exécution : {exec_time:.2f} secondes")