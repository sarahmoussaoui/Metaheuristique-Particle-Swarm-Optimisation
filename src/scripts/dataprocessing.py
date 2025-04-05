import os
import glob


def process_jssp_dataset(input_file, base_dir):
    with open(input_file, "r") as f:
        content = f.readlines()

    problem_count = 0
    i = 0
    while i < len(content):
        line = content[i].strip()
        if line.startswith("Nb of jobs, Nb of Machines"):
            # Start of a new problem
            header1 = line
            i += 1
            header2 = content[i].strip()
            i += 1

            # Parse the numbers
            parts = [p.strip() for p in header2.split()]
            nb_jobs = int(parts[0])
            nb_machines = int(parts[1])
            problem_count += 1

            # Prepare output content
            output_content = []
            output_content.append(header1)
            output_content.append(header2)

            # Process Times section
            if i < len(content) and content[i].strip() == "Times":
                output_content.append("Times")
                i += 1

                times_lines = []
                for _ in range(nb_jobs):
                    if i >= len(content):
                        break
                    line = content[i].strip()
                    if line:  # Only add non-empty lines
                        times_lines.append(f" {line}")  # Add single space before
                    i += 1
                output_content.extend(times_lines)

            # Process Machines section
            if i < len(content) and content[i].strip() == "Machines":
                output_content.append("Machines")
                i += 1

                machines_lines = []
                for _ in range(nb_jobs):
                    if i >= len(content):
                        break
                    line = content[i].strip()
                    if line:  # Only add non-empty lines
                        machines_lines.append(f" {line}")  # Add single space before
                    i += 1
                output_content.extend(machines_lines)

            # Skip any remaining lines until next problem
            while i < len(content) and not content[i].strip().startswith(
                "Nb of jobs, Nb of Machines"
            ):
                i += 1

            # Create output directory if needed
            os.makedirs(base_dir, exist_ok=True)

            # Create output filename
            filename = os.path.join(
                base_dir, f"data_{nb_jobs}j_{nb_machines}m_{problem_count}.txt"
            )

            # Write to file
            with open(filename, "w") as out_f:
                out_f.write("\n".join(output_content))

            print(f"Created {filename}")
        else:
            i += 1


# === MAIN ===
if __name__ == "__main__":
    input_dir = ".\\src\\data\\dataset"
    base_dir = ".\\src\\data\\processed"

    # Get all .txt files in the dataset directory
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    for input_filename in txt_files:
        print(f"Processing {input_filename}...")
        process_jssp_dataset(input_filename, base_dir=base_dir)

    print("Processing complete!")
