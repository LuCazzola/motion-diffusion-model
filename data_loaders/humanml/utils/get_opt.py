import json
from argparse import Namespace
import re
from os.path import join as pjoin
from data_loaders.humanml.utils.word_vectorizer import POS_enumerator

def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = bool(value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'latest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    opt.fewshot = opt.fewshot if hasattr(opt, 'fewshot') else False

    # Few-shot options, they can be omitted
    base_dataset_root = './dataset'
    if opt.dataset_name == 't2m':
        # HumanML3D dataset
        opt.data_root = pjoin(base_dataset_root, 'HumanML3D')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
    elif opt.dataset_name == 'kit':
        # KIT Motion Language dataset
        opt.data_root = pjoin(base_dataset_root, 'KIT-ML')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
    elif opt.dataset_name == 'ntu60':
        # NTU RGB+D dataset
        opt.data_root = pjoin(base_dataset_root, 'NTU60')
        opt.default_data_root = pjoin(opt.data_root, 'splits', 'default')
        opt.fewshot_data_root = pjoin(opt.data_root, 'splits', 'fewshot')
        opt.fewshot_meta_file = 'meta.json' # generation details of the few-shot (if in few-shot mode)
        opt.action_captions = pjoin(opt.data_root, 'class_captions.json') # maps class names to captions

        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196

        if not opt.fewshot:
            opt.data_root = pjoin(opt.default_data_root, opt.task_split)
        else:
            opt.data_root = pjoin(opt.fewshot_data_root, opt.fewshot_id, opt.task_split)
            opt.pretrain_data_root = pjoin(base_dataset_root, opt.pretrain_dataset)
            with open(pjoin(opt.fewshot_data_root, opt.fewshot_id, opt.fewshot_meta_file,), 'r') as f:
                opt.fewshot_metadata = Namespace(**json.load(f))
            
        # NOTE: NTU60 should be treated as t2m dataset, therefore, we overwrite the dataset name
        opt.dataset_name= 't2m' # checking the dataset name has significance in the codebase
    else:
        raise KeyError(f'Dataset "{opt.dataset_name}" not recognized')

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt