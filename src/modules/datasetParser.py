class DatasetParser:
    @staticmethod
    def parse(dataset_str):
        lines = dataset_str.strip().split("\n")

        # Skip metadata/header lines
        i = 0
        while i < len(lines) and not lines[i].strip().startswith("Times"):
            i += 1

        # Go back to get the number of jobs and machines line , upper_bound, lower_bound
        meta_line = lines[i - 1]
        parts = meta_line.split()
        num_jobs, num_machines, upper_bound, lower_bound = (
            int(parts[0]),
            int(parts[1]),
            int(parts[4]),
            int(parts[5]),
        )

        # Now extract time matrix
        i += 1  # Skip "Times"
        times = []
        while i < len(lines) and not lines[i].strip().startswith("Machines"):
            times.append(list(map(int, lines[i].strip().split())))
            i += 1

        # Now extract machine matrix
        i += 1  # Skip "Machines"
        machines = []
        while i < len(lines) and not lines[i].strip().startswith("Nb"):
            machines.append(list(map(int, lines[i].strip().split())))
            i += 1

        return num_jobs, num_machines, upper_bound, lower_bound, times, machines
