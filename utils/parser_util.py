from argparse import ArgumentParser
from argparse import Namespace
import argparse
import os
import json
from copy import deepcopy

LORA_PREFIX = "LoRA"
MOE_PREFIX = "MoE"
ADAPTERS = [
    LORA_PREFIX,
    MOE_PREFIX
]

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    if args.model_path != '':  # if not using external results file
        args = load_args_from_model(args, args_to_overwrite)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    
    return apply_rules(args)

def load_args_from_model(args, args_to_overwrite):
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
    return args

def apply_rules(args):
    # For prefix completion
    if args.pred_len == 0:
        args.pred_len = args.context_len
    # For target conditioning
    if args.lambda_target_loc > 0.:
        args.multi_target_cond = True

    return args

def wrap_adapter_args(args):
    """ Wraps adapter arguments into sub-namespaces for better organization."""
    for PREFIX in ADAPTERS:
        new_args, old_keys = group_args_by_prefix(args, PREFIX)
        setattr(args, PREFIX, new_args) 
        for k in old_keys:
            delattr(args, k)
        setattr(getattr(args, PREFIX), 'finetune', PREFIX in args.peft)
    return args

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def serialize_args(ns):
    if isinstance(ns, Namespace):
        return {k: serialize_args(v) for k, v in vars(ns).items()}
    return ns

def de_serialize_args(d):
    if isinstance(d, dict):
        return Namespace(**{k: de_serialize_args(v) for k, v in d.items()})
    return d

def group_args_by_prefix(args: Namespace, prefix: str):
    """Groups arguments by a given prefix and returns the extracted namespace and list of keys."""
    group_dict = {k[len(prefix)+1:]: v for k, v in vars(args).items() if k.startswith(f"{prefix}_")}
    extracted_keys = list(group_dict.keys())
    # Remember: extracted_keys contain keys after stripping the prefix.
    # You likely want the full keys to delete from args:
    full_keys = [f"{prefix}_{k}" for k in extracted_keys]
    return Namespace(**group_dict), full_keys

def get_opts_from_adapter_name(name: str, args: Namespace):

    adapter_args = getattr(args, name, None)
    assert adapter_args is not None, f"Adapter arguments for {name} were not found in args."

    if name == LORA_PREFIX:
        from modules.lora_pytorch import namespace_to_lora_opt
        return namespace_to_lora_opt(adapter_args)
    elif name == MOE_PREFIX:
        from modules.moe_pytorch import namespace_to_moe_opt
        return namespace_to_moe_opt(adapter_args)
    else :
        raise ValueError(f"Unknown adapter name: {name}. Supported: ['LoRA', 'MoE'].")


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')

def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--external_mode", default=False, type=bool, help="For backward cometability, do not change or delete.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str, help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int, help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")

def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--text_encoder_type", default='clip',
                       choices=['clip', 'bert'], type=str, help="Text encoder type.")
    group.add_argument("--emb_trans_dec", action='store_true',
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--mask_frames", action='store_true', help="If true, will fix Rotem's bug and mask invalid frames.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_target_loc", default=0.0, type=float, help="For HumanML only, when . L2 with target location.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--pos_embed_max_len", default=5000, type=int,
                       help="Pose embedding max length.")
    group.add_argument("--use_ema", action='store_true',
                    help="If True, will use EMA model averaging.")
    
    group.add_argument("--multi_target_cond", action='store_true', help="If true, enable multi-target conditioning (aka Sigal's model).")
    group.add_argument("--multi_encoder_type", default='single', choices=['single', 'multi', 'split'], type=str, help="Specifies the encoder type to be used for the multi joint condition.")
    group.add_argument("--target_enc_layers", default=1, type=int, help="Num target encoder layers")

    # Prefix completion model
    group.add_argument("--context_len", default=0, type=int, help="If larger than 0, will do prefix completion.")
    group.add_argument("--pred_len", default=0, type=int, help="If context_len larger than 0, will do prefix completion. If pred_len will not be specified - will use the same length as context_len")
    

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc', 'ntu60'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    #group.add_argument("--split", default='train', type=str,
    #                      help="Which split to train on.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=1e-5, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='val', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=1, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=None, type=int,
                       help="If None, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=25, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=250, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=10_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--starting_checkpoint", default="", type=str)
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=3, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=2, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--gen_guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    
    group.add_argument("--avg_model_beta", default=0.9999, type=float, help="Average model beta (for EMA).")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")
    
    group.add_argument("--target_joint_names", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")


def add_peft_options(parser): # Parameter Efficient Fine-Tuning options [LoRA, MoE, etc.]
    group = parser.add_argument_group('peft')
    group.add_argument("--peft", nargs='*', type=str, default=[], choices=ADAPTERS, help="Type of PEFT to use. (LoRA, MoE, both, ...). ")
    # LoRA options
    group.add_argument(f"--{LORA_PREFIX}_where", nargs='*', type=str, default=["transformer"], choices=['transformer', 'conditioning', 'denoising_head'], help="Where to apply LoRA within MDM.")
    group.add_argument(f"--{LORA_PREFIX}_tLayer", nargs='*', type=int, default=[-1], help="Transformers layer to use for lora, -1 for all layers.")
    group.add_argument(f"--{LORA_PREFIX}_rank", default=5, type=int, help="Rank of the LoRA layers.")
    group.add_argument(f"--{LORA_PREFIX}_ff", action='store_true', help="add LoRA to the feed-forward layers.")
    group.add_argument(f"--{LORA_PREFIX}_no_q", action='store_true', help="remove LoRA adapter from query.")
    # MoE options
    group.add_argument(f"--{MOE_PREFIX}_where", nargs='*', type=str, default=["transformer"], choices=['transformer', 'conditioning', 'denoising_head'], help="Where to apply MoE within MDM.")
    group.add_argument(f"--{MOE_PREFIX}_tLayer", nargs='*', type=int, default=[-1], help="Transformers layer to use for MoE, -1 for all layers.")
    group.add_argument(f"--{MOE_PREFIX}_num_experts", default=5, type=int, help="Number of experts in the MoE layer.")
    group.add_argument(f"--{MOE_PREFIX}_num_experts_per_tok", default=2, type=int, help="Number of experts per token in the MoE layer.")
    group.add_argument(f"--{MOE_PREFIX}_gate_type", default='linear', type=str, choices=['linear'], help="Type of the gate module in the MoE layer.")
    group.add_argument(f"--{MOE_PREFIX}_gate_bias", action='store_true', help="If true, will use bias in the gate module of the MoE layer.")
    group.add_argument(f"--{MOE_PREFIX}_routing_strategy", default='topk', type=str, choices=['topk'], help="Routing strategy for the MoE layer.")
    group.add_argument(f"--{MOE_PREFIX}_lora_experts", action='store_true', help="If true, will use LoRA instead of FF.")
    group.add_argument(f"--{MOE_PREFIX}_lora_experts_rank", default=5, type=int, help="Rank of the LoRA experts.")

def add_training_fewshot_options(parser):
    group = parser.add_argument_group('few_shot_training')
    #group.add_argument("--few_shot", action='store_true', help="If true, few-shot generation is assumed.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=6, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")

    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")

def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--dynamic_text_path", default='', type=str,
                       help="For the autoregressive mode only! Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--target_joint_names", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")

def add_generate_aug_options(parser):
    # Augmentation applied to the generated motion
    group = parser.add_argument_group('generate_aug')
    group.add_argument("--motion_length_noise", type=float, default=1.5,
                            help="variation (in senconds) applied to the generated motion.")

def add_rendering_options(parser):
    # Options associated to animation rendering into .mp4 files
    group = parser.add_argument_group('rendering')
    group.add_argument("--no_render", action='store_true',
                        help="If set, will not render the generated motions.")
    group.add_argument("--freeze_uneven_anim", action='store_true',
                        help="If set, will freeze the uneven animation. Otherwise it will be trimmed")

def add_generate_t2m_action_options(parser):
    # Hybrid generation mode for the few-shot synthesis of actions
    # from text description instead of class labels
    group = parser.add_argument_group('t2m_action_gen')
    group.add_argument("--t2m_action_gen", action='store_true',
                        help="If true, few-shot generation is assumed.")
    group.add_argument("--action_labels", nargs='+', type=int, default=[],
                        help="List of action labels (indexes) for few-shot generation.")
    group.add_argument("--action_captions", default='', type=str,
                        help="Path to a .json file listing viable natural language descriptions per action class")

def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml', 'ntu60']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_training_fewshot_options(parser)
    add_peft_options(parser)

    return wrap_adapter_args(apply_rules(parser.parse_args()))


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_rendering_options(parser)
    add_generate_aug_options(parser)
    add_generate_t2m_action_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)