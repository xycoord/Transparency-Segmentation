import torch
import yaml
import argparse

def create_parser_from_yaml(yaml_path):
    type_map = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool
    }
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    for group_name, args in config.items():
        group = parser.add_argument_group(group_name)
        for arg_name, arg_config in args.items():
            if 'type' in arg_config and isinstance(arg_config['type'], str):
                arg_config['type'] = type_map.get(arg_config['type'])
            group.add_argument(
                f'--{arg_name.replace("_", "-")}',
                **arg_config
            )
    
    return parser

def add_dtype_argument(args):
    dtype_map = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32
    }
    
    if args.data_type in dtype_map:
        args.torch_dtype = dtype_map[args.data_type]
    else:
        raise ValueError(f"Unknown data type: {args.data_type}")
    return args

def parse_args():
    parser = create_parser_from_yaml('config.yaml')
    args = parser.parse_args()
    args = add_dtype_argument(args)
    return args