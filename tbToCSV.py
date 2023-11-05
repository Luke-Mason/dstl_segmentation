import os
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
from tqdm import tqdm
import sys

def to_df(dpath):
    # Get the train and validation summaries toigether into a list of tuples

    summary_iterators = [EventAccumulator(os.path.join(dpath, dname,subdir)).Reload() for dname in os.listdir(dpath) for subdir in os.listdir(os.path.join(dpath, dname))]
    data = []
    for summary in summary_iterators:
        # Check path for if it is a val or train
        # summary.path
        tags = summary.Tags()['scalars']
        # Get the name of the column
        tail, head = os.path.split(summary.path)
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

    return combined

def tbToCsv(mainPath: str):
    allPaths = list(glob.iglob(mainPath+'/**/**/**/*', recursive=True))
    print("Converting TB Events to CSV:")

    data = []
    for i in tqdm(range(0,len(allPaths),1)):
        fileName = allPaths[i]
        try:
            df = to_df(fileName)
            data.append(df)
        except Exception as e:
            print("Error with file:",fileName)
            print(str(e))
            break

    combined = None
    for df in data:
        if combined is None:
            combined = df
        else:
            combined = pd.merge(combined, df, on='step', how='outer')

    if combined is not None:
        combined.to_csv(f"{mainPath}.tsv", index=False, sep='\t')

if __name__ == '__main__':
    run_name = sys.argv[1]
    tbToCsv(f"saved/runs/dstl_ex{run_name}")
