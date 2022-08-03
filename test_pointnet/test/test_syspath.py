import os
src_dir = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__))
# if not src_dir.endswith("sfa"):
src_dir = os.path.dirname(src_dir)
print(src_dir)
# if src_dir not in sys.path:
#     sys.path.append(src_dir)