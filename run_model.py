import time
import torch
import torch.backends.cudnn as cudnn
import dataloaders
import models
import json
import os
from argparse import ArgumentParser



def compute_speed(model, input_size, device, iteration=1000):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    for _ in range(50):
        model(input)

    print('=========Eval Forward Time=========')
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))

    description = str(iteration) + ',' + str(input_size[3]) + ',' + str(elapsed_time) + ',' \
                + str(speed_time) + ',' + str(fps)
    logFileLoc = 'forward_time.txt'
    print(logFileLoc)
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write('Iterations,Input_size, Elapsed_Time(s/100_%d iters),Speed_Time(ms/iter),FPS'% (iteration) + '\n')
    logger.write(description + '\n')

    return speed_time, fps


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--size", type=str, default="256,256", help="input size of model")
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--iter', type=int, default=1000)
    #parser.add_argument('--model', type=str, default='UFONet')
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    config = json.load(open(args.config))

    # h, w = map(int, args.size.split(','))
    # loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    # num_classes = loader.dataset.num_classes
    # model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    # compute_speed(model, (args.batch_size, args.num_channels, h, w), int(args.gpus), iteration=args.iter)

    loader = DSTLLoader(
        **all_loader_config,
        files=train_patch_files,
        weights=train_patch_weights,
        **config["train_loader"]["args"]
    )



    dstl_data_path = os.environ.get('DSTL_DATA_PATH')
    if dstl_data_path is None:
        raise EnvironmentError(
            'DSTL_DATA_PATH environment variable is not set, '
            'it must be a path to your DSTL data directory.')

    # Load the CSV into a DataFrame
    df = pd.read_csv(
        os.path.join(dstl_data_path, 'train_wkt_v4.csv/train_wkt_v4.csv'))

    # Get the data metadata list.
    _wkt_data = {}

    row = df.iloc[12]
    im_id = row['ImageId']
    class_type = row['ClassType']
    poly = row['MultipolygonWKT']

    # Add the polygon to the dictionary
    _wkt_data.setdefault(im_id, {})[int(class_type)] = poly

    _wkt_data = list(_wkt_data.items())

    training_classes_ = config['all_loader']['preprocessing']['training_classes']

    # Stratified K-Fold
    mask_stats = json.loads(Path(
        'dataloaders/labels/dstl-stats.json').read_text())
    image_ids = df['ImageId'].unique()


    sorted_by_area = sorted(im_area, key=lambda x: str(x[1]), reverse=True)
    sorted_by_area = [t[0] for t in sorted_by_area]
    logger.debug(f"Sorted Area {sorted_by_area}")
    split_count = config["trainer"]["k_split"] if len(im_area) > \
                                                  config["trainer"][
                                                      "k_split"] else len(
        im_area)
    arr = stratified_split(sorted_by_area, split_count)
    stratisfied_indices = arr.flatten()

    # LOSS
    loss = getattr(losses, config['loss'])(threshold=config['threshold'])
    start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')

    if config["trainer"]["val"]:
        # Split the data into K folds
        shuffle_ = config["trainer"]["k_shuffle"]
        random_state_ = config["trainer"][
            "k_random_state"] if shuffle_ else None
        kfold = KFold(n_splits=split_count, shuffle=shuffle_,
                      random_state=random_state_)

        area_by_id = dict(im_area)

        preprocessing_ = config['all_loader']['preprocessing']
        training_classes_str = '_'.join(
            str(i) for i in preprocessing_['training_classes'])
        training_band_groups = [f"({'_'.join(map(str, band_group['bands']))})"
                                for band_group in
                                preprocessing_['training_band_groups']]
        run_name = (f"batch_size_{config['all_loader']['args']['batch_size']}"
                    f"_lr_{config['optimizer']['args']['lr']}"
                    f"_k_stop_{str(config['trainer']['k_stop'])}"
                    f"_epochs_{config['trainer']['epochs']}"
                    f"_loss_{config['loss']}"
                    f"_scheduler_{config['lr_scheduler']['type']}"
                    f"_patch_size_{preprocessing_['patch_size']}"
                    f"_overlap_pixels_{preprocessing_['overlap_pixels']}"
                    f"_training_classes_({training_classes_str})"
                    f"_training_band_groups_[{'-'.join(training_band_groups)}]")
        writer_dir = os.path.join(config['trainer']['log_dir'], config['name'],
                                  run_name, start_time)
        writer = tensorboard.SummaryWriter(writer_dir)

        train_indxs = stratisfied_indices[train_indxs_of_indxs]
        val_indxs = stratisfied_indices[val_indxs_of_indxs]

        preprocessor = get_preprocessor(config, dstl_data_path,
                                        _wkt_data, start_time,
                                        train_indices=train_indxs,
                                        val_indices=val_indxs)

        train_patch_files, val_patch_files = preprocessor.get_files()

        logger.info("Creating file weights..")
        train_patch_weights, val_patch_weights = preprocessor.get_file_weights()

        # Check if train or val weights contains Inf or NaN
        if np.any(np.isnan(train_patch_weights)) or np.any(
                np.isnan(val_patch_weights)):
            logger.error("Train or val weights contains Inf")
            bonus += 1
            continue

        # Create train and valiation data loaders that only load the data
        # into batch by seleecting indexes from the list of indices we
        # give each loader.
        all_loader_config = {
            "batch_size": config["all_loader"]["args"]["batch_size"],
            "num_workers": config["all_loader"]["args"]["num_workers"],
            "return_id": config["all_loader"]["args"]["return_id"],
        }
