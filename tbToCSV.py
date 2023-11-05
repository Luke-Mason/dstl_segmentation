import os
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
from tqdm import tqdm
import sys
import re

def to_df(date_path):

    # check if the path is a directory
    if not os.path.isdir(date_path):
        return None

    event_paths = []
    for metric in os.listdir(date_path):
        if os.path.isdir(os.path.join(date_path, metric)):
            event_paths += [os.path.join(date_path, metric, file_name) for file_name in
                        os.listdir(os.path.join(date_path, metric))]

    # Get the train and validation summaries toigether into a list of tuples
    summary_iterators = [EventAccumulator(event_path).Reload() for event_path in event_paths]
    data = []
    for summary in summary_iterators:
        # Check path for if it is a val or train
        # summary.path
        tags = summary.Tags()['scalars']
        # Get the name of the column
        tail, head = os.path.split(summary.path)
        tail, head = os.path.split(tail)

        frame = {
            "step": [],
            head: []
        }
        for tag in tags:
            for event in summary.Scalars(tag):
                frame[head].append(event.value)
                frame['step'].append(event.step)
        df = pd.DataFrame(frame, index=frame['step'], columns=['step', head])
        data.append(df)

    combined = None
    for df in data:

        if combined is None:
            combined = df
        else:
            combined = pd.merge(combined, df, on='step', how='outer')

    negatives_df = combined.filter(regex='Negative.*')
    loss_df = combined.filter(regex='All_Loss.*|All_Lr_0_train')
    # Strip All_ from names
    loss_df.columns = [col[len("All_"):] for col in loss_df.columns]

    regex_pattern = r'All.*|Negative.*|csv.*|event.*'
    combined = combined.filter(regex=f"^(?!{regex_pattern}$)")
    # Step is the first column, object names column is the second onwards
    object_name = combined.columns.tolist()[1].split('_')[0]

    # Rename columns
    negatives_df.columns = [f"{object_name}_{col}" for col in negatives_df.columns]
    loss_df.columns = [f"{object_name}_{col}" for col in loss_df.columns]

    combined = pd.concat([loss_df, combined, negatives_df], axis=1)

    combined.columns = [col.replace(" ", "_") for col in combined.columns]
    return combined

def tbToCsv(mainPath: str):
    allPaths = list(glob.iglob(mainPath+'/*/*', recursive=True))
    print("Converting TB Events to CSV:")

    data = []
    for i in tqdm(range(0,len(allPaths),1)):
        fileName = allPaths[i]
        df = to_df(fileName)
        if df is not None:
            data.append(df)

    combined = pd.concat(data, axis=1)

    # Remove steps column and add it back as an index of row num
    combined = combined.filter(regex='^(?!step$)')
    combined['step'] = range(len(combined))
    combined = combined.set_index('step')
    combined.to_csv(f"{mainPath}.tsv", index=True, sep='\t')

if __name__ == '__main__':
    run_name = sys.argv[1]
    tbToCsv(f"saved/runs/dstl_ex{run_name}")
