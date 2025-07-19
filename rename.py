import os
import argparse

parser = argparse.ArgumentParser(description='Renaming image frames')
parser.add_argument('--directory', dest='directory', required=True)
args = parser.parse_args()

def rename_images(directory):
    marker = 0
    for i in range (0, 304):
        name0 = f"frame{i}_0.333_{i+1}.jpg"
        name1 = f"frame{i}_0.666_{i+1}.jpg"
        name2 = f"frame{i+1}.jpg"
        old_names = [name0, name1, name2]

        name0_new = f"frame{marker+1}new.jpg"
        name1_new = f"frame{marker+2}new.jpg"
        name2_new = f"frame{marker+3}new.jpg"
        new_names = [name0_new, name1_new, name2_new]

        marker = marker + 3

        for n in range(3):
            filename = old_names[n]
            new_name = new_names[n]
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_name)
            print(f"Renaming {filename} -> {new_name}")
            os.rename(src, dst)

    for i in range (1, 913):
        filename = f"frame{i}new.jpg"
        new_name = f"frame{i}.jpg"
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_name)
        print(f"Renaming {filename} -> {new_name}")
        os.rename(src, dst)

rename_images(args.directory)