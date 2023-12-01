import argparse
from rotation import rotation
from mesh import mesh

parser = argparse.ArgumentParser()
parser.add_argument("--merged_path", type=str)
parser.add_argument("--rotated_path", type=str)
parser.add_argument("--mesh_path", type=str)
parser.add_argument("--alpha", type=float, default = 5e-4)
parser.add_argument("--margin", type=float, default = 10)
parser.add_argument("--res", type=float, default = 0.3)
args = parser.parse_args()

rotation(args.rotated_path, args.merged_path, args.alpha)
mesh(args.rotated_path, args.mesh_path, args.margin, args.res)