import os
import shutil


def split_dataset_into_3(path_to_dataset, train_path, train_ratio, valid_ratio):
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))  # retrieve name of subdirectories
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)
    print(sub_dirs)
    print(sub_dir_item_cnt)

    dir_train = os.path.join(os.path.dirname(train_path), 'train')
    dir_valid = os.path.join(os.path.dirname(train_path), 'validation')

    for i, sub_dir in enumerate(sub_dirs):
        # """print sub-directory 0, 1, 2... inside train and validation directory"""
        dir_train_dst = os.path.join(dir_train, sub_dir)  # directory for destination of train dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)  # directory for destination of validation dataset
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))
        #     """items will list out the content of the sub-dir (i.e. in jpg format)"""
        items = os.listdir(sub_dir)
        print(sub_dir_item_cnt[i])

        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio),
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        for item_idx in range(round(sub_dir_item_cnt[i])):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)
                source_file = os.path.join(sub_dir, items[item_idx])
                dst_file = os.path.join(dir_valid_dst, items[item_idx])
                shutil.copyfile(source_file, dst_file)

    return


original_dataset_path = './sub/'
train_dataset_path = './Module8Testing'
tr_ratio = 0.9
vl_ratio = 0.1
split_dataset_into_3(original_dataset_path, train_dataset_path, tr_ratio, vl_ratio)


