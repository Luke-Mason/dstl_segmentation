import os
import re

def main():

    # saved/runs/dstl_ex<exp>/<idx>/0-<name>
    # Read the directories in saved/runs
    experiements = os.listdir('saved/runs')

    # Get the directories in each of those experiments
    for exp_dir in experiements:
        # Check exp_dir is a directory and not a file
        if not os.path.isdir('saved/runs/' + exp_dir):
            continue
        if not exp_dir.startswith('dstl_ex'):
            continue
        class_runs = os.listdir('saved/runs/' + exp_dir)
        for class_run_dir in class_runs:
            # Get the i index from class_run_dir from the name part
            # training_classes(x)
            pattern = r'training_classes_\((\d+)\)'

            # Use re.search to find the pattern in the input string
            match = re.search(pattern, class_run_dir)

            # Check if a match was found
            if match:
                # Extract the matched number from the regex group
                i = match.group(1)
            else:
                raise ValueError(f'No match found for {pattern} in {class_run_dir}')

            attempts = os.listdir('saved/runs/' + exp_dir + '/' + class_run_dir)
            if len(attempts) < 1:
                raise ValueError(f'Expected 1 attempt in saved/runs/{exp_dir}'
                                 f'/{class_run_dir}, found'
                       f' {len(attempts)}')

            attempts.sort()
            dir_name = attempts[-1]
            if len(os.listdir('saved/runs/' + exp_dir + '/' + class_run_dir
                               + '/' + dir_name)) < 5:
                raise ValueError(f'Expected at least 5 files in '
                                 f'saved/runs/{exp_dir}/{class_run_dir}'
                                 f'/{dir_name}, found')

            # Get the experiement number from run name
            exp = exp_dir.split('_')[1]

            # Remove first 2 characters
            exp = exp[2:]

            # cp saved/checkpoints/dstl_ex<exp>/<name>/K_0/ saved/models/ex<exp>/<idx>
            checkpoint = f'saved/checkpoints/{exp_dir}/{dir_name}/K_0/'
            if os.path.exists(checkpoint):
                files = os.listdir(checkpoint)
                # Check files is not empty
                if not files:
                    raise ValueError(f'No files in {checkpoint}')

                # get best_model.pth or if it does not exist, sort the files
                # by file name and get the largest filename in the list
                if 'best_model.pth' in files:
                    checkpoint += 'best_model.pth'
                else:
                    files.sort()
                    checkpoint += files[-1]

                # Create the models directory if it does not exist
                models_dir = f'saved/models/ex{exp}/{i}'
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)

                # Copy the checkpoint to the models directory
                os.system(f'cp {checkpoint} {models_dir}/best_model.pth')

if __name__ == '__main__':
    main()