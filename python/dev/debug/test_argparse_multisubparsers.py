# multiple subparsers doesn't work

import argparse

p = argparse.ArgumentParser(description='Test parser 1')
p.add_argument('--arg1', type=int, help='Argument 1')

p_model = p.add_subparsers(title='Model', dest='model', required=True, metavar="MODEL")
p_model_a = p_model.add_parser('a', help='Model A')
p_model_b = p_model.add_parser('b', help='Model B')

p_model_a.add_argument('--arga1', type=int, help='Argument a1')
p_model_a.add_argument('--arga2', type=int, help='Argument a2')

p_model_b.add_argument('--argb1', type=int, help='Argument b1')

p_data = p.add_subparsers(title='Data', dest='data', required=True, metavar="DATA")
p_data_a = p_data.add_parser('x', help='Data X')
p_data_b = p_data.add_parser('y', help='Data Y')

p_data_a.add_argument('--argx1', type=int, help='Argument x1')
p_data_a.add_argument('--argx2', type=int, help='Argument x2')

p_data_b.add_argument('--argy1', type=int, help='Argument y1')

argvs = [
    "--help",
    # "a --help",
    # "b --help",
]
for argv in argvs:
    argv = argv.split()
    print(f"argv={argv}")
    try:
        args = p.parse_args(argv)
        print(f"args: {args}")
    except Exception as ex:
        print(f"exception: {ex}")
        
        