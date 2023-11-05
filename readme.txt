UFONet training:
select class in coco_1.txt and get index. For example, 24 giraffe. 
change ID_TO_TRAINID dict. For example, 24:1, others 0.
change root_image and root_label path to folder of class name in data_prepare.py to get train val test split
in data/coco/giraffe/images folder, move images in test folder to train folder
change config.json name, data_dir, batch_size, crop_size
run python train.py

UFONet evaluation:
inference_bbox.py change path_model, h, folder and dict_label. run inference_bbox.py
coco_get_bbox.py change folder, cls_list, cls = cat_names[anns[i]['category_id']-2] . run coco_get_bbox.py
get_map_origin.py change DR_PATH. run get_map_origin.py
change results folder name


YOLOv4-tiny training:
coco_get_bbox.py change folder, cls_list, cls = cat_names[anns[i]['category_id']-2]. run coco_get_bbox.py
voc_annotation.py change classes, folder. run voc_annotation.py. check new generated txt. for example giraffe.txt
train.py change folder name logs_, classes_path, loss_history = LossHistory("logs_giraffe/"), annotation_path = 'giraffe.txt'
in folder model_data create giraffe.txt
run python train.py

YOLOv4-tiny evaluation:
coco_get_bbox.py change folder = 'giraffe', cls_list = ['giraffe']. run coco_get_bbox.py
get_dr_txt.py change folder. run get_dr_txt.py
get_map_origin.py change GT_PATH. run get_map_origin.py
change folder names of input and results









