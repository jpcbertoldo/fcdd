import argparse

p1 = argparse.ArgumentParser(description='Test parser 1')
p1.add_argument('--arg1', type=int, help='Argument 1')

p2 = argparse.ArgumentParser(description='Test parser 2')
p2.add_argument('--arg2', type=int, help='Argument 2')

# case 00
# argv = "--arg1 1 --arg2 2"
# argv = argv.split()
# args1 = p1.parse_args(argv)
# breaks because --arg2 is unknown

# case 01
# argv = "--arg1 1 --arg2 2"
# argv = argv.split()
# args1, argv = p1.parse_known_args(argv)
# args2, argv = p2.parse_known_args(argv)
# print(f"args1: {args1}")
# print(f"args2: {args2}")
# works perfectly

# case 02
# argv = "--arg1 1 --arg2 2"
# argv = argv.split()
# args1, argv = p1.parse_known_args(argv)
# args2 = p2.parse_args(argv)
# print(f"args1: {args1}")
# print(f"args2: {args2}")
# works perfectly

# case 03
# argv = "--arg1 1"
# argv = argv.split()
# args1, argv = p1.parse_known_args(argv)
# args2 = p2.parse_args(argv)
# print(f"args1: {args1}")
# print(f"args2: {args2}")
# works perfectly
# args2 gets None by default default

# case 04
# p1.add_argument('--arg3', type=int, help='Argument 3', required=True)
# p2.add_argument('--arg3', type=int, help='Argument 3', required=True)
# argv = "--arg1 1 --arg2 2"
# argv = argv.split()
# args1, argv = p1.parse_known_args(argv)
# args2 = p2.parse_args(argv)
# print(f"args1: {args1}")
# print(f"args2: {args2}")
# breaks because --arg3 is required

# case 05
# argv = "--arg1 1 --arg2 2 --arg3 3"
# argv = argv.split()
# args1, argv = p1.parse_known_args(argv)
# args2 = p2.parse_args(argv)
# print(f"args1: {args1}")
# print(f"args2: {args2}")
# # breaks unrecognized argument

# case 06: dest default?
# p1.add_argument("--argA", dest="A", type=int, help="Argument A")
# p1.set_defaults(A=10)
# argv = "--arg1 1 --arg2 2"
# argv = argv.split()
# args1, argv = p1.parse_known_args(argv)
# args2 = p2.parse_args(argv)
# print(f"args1: {args1}")
# print(f"args2: {args2}")

# case 07: default without arg?
p1.set_defaults(A=10, B=20)
argv = "--arg1 1 --arg2 2"
argv = argv.split()
args1, argv = p1.parse_known_args(argv)
args2 = p2.parse_args(argv)
print(f"args1: {args1}")
print(f"args2: {args2}")