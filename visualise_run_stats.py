import tbToCSV as tb
import pandas as pd
import os
from utils import metric_indx

def visualise_run_stats(run_path: str, run_desc: str):
    df = pd.read_csv(f"{run_path}.tsv", sep='\t')
    # Loop over each object
    for object_name in metric_indx.values():
        for metric in ["Mean_IoU", "Loss", "F1"]:
            for run in ["train", "val"]:
                # Get the columns for the object
                cols = df.filter(regex=f".*{object_name}.*_{metric}"
                                       f".*_{run}.*").columns
                # Get the columns for the object
                df[cols].plot(title=f"{run_desc} {object_name} {metric} {run}")
                plt.show()
                plt.close()

if __name__ == '__main__':

    runs = {
        "1-1": "Bands 1-3",
        # "2-1": "Bands 2-4",
        "3-1": "Bands 3-5",
        "4-1": "Bands 4-6",
        "5-1": "Bands 5-7",
        "6-1": "Bands 6-8",
        "7-1": "Bands 7-9",
        "8-1": "Bands 8-10",
        "9-1": "Bands 9-11",
        "10-1": "Bands 10-12",
        "11-1": "Bands 11-13",
        "12-1": "Bands 12-14",
    }
    for run_id, run_desc in runs.items():
        run_path = f"saved/runs/dstl_ex{run_id}"

        # Check run has already been processed
        if not os.path.exists(f"{run_path}.tsv"):
            tb.tbToCsv(run_path)

        visualise_run_stats(run_path, run_desc)