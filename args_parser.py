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

def parse_args():
    parser = create_parser_from_yaml('config.yaml')
    args = parser.parse_args()
    return args