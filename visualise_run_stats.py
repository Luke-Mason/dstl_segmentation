import tbToCSV as tb
import pandas as pd
import os
from utils import metric_indx
import matplotlib.pyplot as plt
import seaborn as sns

obj_colors = {
    "Buildings": "#FF0000",         # Red
    "Misc": "#FFA500",              # Orange
    "Road": "#CCCC00",              # Yellow
    "Track": "#00C900",             # Green
    "Trees": "#0000FF",             # Blue
    "Crops": "#800080",             # Purple
    "Waterway": "#7200DC",          # Blue Violet
    "Standing_water": "#00FFFF",    # Cyan
    "Vehicle_Large": "#D80075",     # Deep Pink
    "Vehicle_Small": "#B40000",     # Fire Brick
    "Negative": "#808080"           # Gray
}

ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
       'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', "x", "y", "z"]
window = 10  # Adjust the window size for smoothing


def visualise_run_stats(id, run_path: str, run_desc: str, dir: str):
    metric = "Pixel_Accuracy"
    df = pd.read_csv(f"{run_path}.tsv", sep='\t')
    df = df.rename(columns={"step": "Epochs"})
    df = df.set_index("Epochs")
    # Loop over each object
    objects = metric_indx.values()
    # Remove "Negative" element
    objects = [item for item in objects if item != "Negative" and item != "All"]
    for i, object_name in enumerate(objects):
        # Replace space with _
        object_name = object_name.replace(" ", "_")

        # Get the columns for the object
        col = df.filter(regex=f".*{object_name}_{metric}.*_"
                              f"(train|val).*").columns.tolist()
        neg_col = df.filter(regex=f".*{object_name}_Negative"
                                  f"_{metric}.*_"
                                  f"(train|val).*").columns.tolist()
        cols = col + neg_col
        if len(cols) == 0:
            continue

        #%%
        # Get the columns for the object
        sns.set(style="whitegrid")

        labels = [f"{object_name} Train", f"{object_name} Val", "Negative Train", "Negative Val"]

        # Specify the DataFrame, columns, and title for the Seaborn plot
        for j, col in enumerate(cols):
            df[col] = df[col].rolling(window=window, min_periods=1).mean()
            linestyle = 'solid' if j % 2 == 0 else 'dashed'
            custom_palette = [color for color in [obj_colors[object_name], obj_colors["Negative"]]
                                                  for _ in range(2)]
            sns.set_palette(custom_palette)
            sns.lineplot(data=df, x="Epochs", y=col, label=labels[j], linestyle=linestyle)

        # plt.title(f"{ids[id]}-{i + 1}. {object_name} - {metric} - {run_desc}", fontsize=24)
        plt.xlabel("Epochs", fontsize=24)
        plt.ylabel(metric, fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{dir}/{ids[id]}-{i + 1}.png')
        plt.close()

        #%%

    # Do the same for the loss of all objects not for each
    all_cols = df.filter(regex=f".*Lr.*").columns.tolist()
    sns.set(style="whitegrid")
    custom_palette = list(obj_colors.values())
    sns.set_palette(custom_palette)

    # Specify the DataFrame, columns, and title for the Seaborn plot
    for j, col in enumerate(all_cols):
        df[col] = df[col].rolling(window=window, min_periods=1).mean()
        sns.lineplot(data=df, x="Epochs", y=col, label=objects[j])
    # plt.title(f"{ids[id]}-{len(objects) + 1}. Learning Rates - {run_desc} ", fontsize=18)
    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Learning Rate", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{dir}/{ids[id]}-{len(objects) + 1}.png')
    plt.close()

    all_cols = []
    labels= []
    metric = "Loss"
    sns.set(style="whitegrid")
    custom_palette = [item for item in obj_colors.values() for _ in range(2)]
    sns.set_palette(custom_palette)
    for i, object_name in enumerate(objects):
        # Get the columns for the object
        col = df.filter(regex=f".*{object_name}_{metric}.*_"
                              f"(train|val).*").columns.tolist()
        neg_col = df.filter(regex=f".*{object_name}_Negative"
                                  f"_{metric}.*_"
                                  f"(train|val).*").columns.tolist()
        cols = col + neg_col
        if len(cols) == 0:
            continue

        all_cols += cols

        labels += [f"{object_name} Train", f"{object_name} Val"]

    # Specify the DataFrame, columns, and title for the Seaborn plot
    for j, col in enumerate(all_cols):
        df[col] = df[col].rolling(window=window, min_periods=1).mean()
        linestyle = 'solid' if j % 2 == 0 else 'dashed'
        ax = sns.lineplot(data=df, x="Epochs", y=col, label=labels[j],
                          linestyle=linestyle)
        ax.set_ylim(-4, 4)
    # plt.title(f"{ids[id]}-{len(objects) + 2}. Losses - {run_desc} ", fontsize=18)
    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Loss", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{dir}/{ids[id]}-{len(objects) + 2}.png')
    plt.close()


def collect_and_visualise(runs, dir):
    for i, (run_id, run_desc) in enumerate(runs.items()):
        run_path = f"saved/runs/dstl_ex{run_id}"

        # Check run has already been processed
        if not os.path.exists(f"{run_path}.tsv"):
            tb.tbToCsv(run_path)

        visualise_run_stats(i, run_path, run_desc, dir)


if __name__ == '__main__':
    dir = "ex_set_1_pa"
    if not os.path.exists(dir):
        os.makedirs(dir)
    runs = {
        "1-1": "Bands 1-3",
        "2-1": "Bands 2-4",
        "3-1": "Bands 3-5",
        "4-1": "Bands 4-6",
        "5-1": "Bands 5-7",
        "6-1": "Bands 6-8",
        "7-1": "Bands 7-9",
        "8-1": "Bands 8-10",
        "9-1": "Bands 9-11",
        "10-1": "Bands 10-12",
        "11-1": "Bands 11-13",
        "12-1": "Bands 12-14"
    }

    # dir = "ex_set_1_f1"
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    # runs = {
    #     "19-1": "Mean Bands (13_14_15_16_3_4)-(1_2_9_10_11_12)-("
    #             "5_6_7_8_17_18_19_20)",
    #     "19-2": "Min Bands (13_14_15_16_3_4)-(1_2_9_10_11_12)-("
    #             "5_6_7_8_17_18_19_20)",
    #     "19-3": "Max Bands (13_14_15_16_3_4)-(1_2_9_10_11_12)-("
    #             "5_6_7_8_17_18_19_20)",
    #     "19-4": "Sum Bands (13_14_15_16_3_4)-(1_2_9_10_11_12)-("
    #             "5_6_7_8_17_18_19_20)",
    #     "20-1": "Mean Bands (1_2_17_18_19_20)-(3_5_6_7_8_9)-("
    #             "4_13_14_15_16_10_11_12)",
    #     "20-2": "Min Bands (1_2_17_18_19_20)-(3_5_6_7_8_9)-("
    #             "4_13_14_15_16_10_11_12)",
    #     "20-3": "Max Bands (1_2_17_18_19_20)-(3_5_6_7_8_9)-("
    #             "4_13_14_15_16_10_11_12)",
    #     "20-4": "Sum Bands (1_2_17_18_19_20)-(3_5_6_7_8_9)-("
    #             "4_13_14_15_16_10_11_12)",
    #     "21-1": "Mean Bands (1_2_3_4)-(5_6_7_8_9_10_11_12)-("
    #             "13_14_15_16_17_18_19_20)",
    #     "21-2": "Min Bands (1_2_3_4)-(5_6_7_8_9_10_11_12)-("
    #             "13_14_15_16_17_18_19_20)",
    #     "21-3": "Max Bands (1_2_3_4)-(5_6_7_8_9_10_11_12)-("
    #             "13_14_15_16_17_18_19_20)",
    #     "21-4": "Sum Bands (1_2_3_4)-(5_6_7_8_9_10_11_12)-("
    #             "13_14_15_16_17_18_19_20)",
    #
    # }

    collect_and_visualise(runs, dir)