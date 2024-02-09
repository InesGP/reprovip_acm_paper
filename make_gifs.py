
import glob
from PIL import Image
from nilearn import plotting
import os
import nibabel as nib
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='Create gifs for IEEE and MCA')
  parser.add_argument('--base', type=str, default='/home/ine5/projects/rrg-glatard/brainhack-2023-linear-registration/results/anat-12dofs/mca', help='Base directory containing nifti files of subjects')
  parser.add_argument('--dof', type=int, help='Degrees of freedom the operation had')
  parser.add_argument('--save-filepath', type=str, help='File directory to where to save gifts')
  return parser.parse_args()


def make_gifs(base, dof, save_filepath):
    subs = [x.split('.')[0] for x in os.listdir(f'{base}/2') if 'gz' in x]

    # print(subs)
    for sub in subs:
        # files = glob.glob(f"/home/ine5/projects/rrg-glatard/brainhack-2023-linear-registration/results/mca/*/{sub}.nii.gz")
        if os.path.isdir(f'{save_filepath}/gifs/anat{dof}dof/{sub}'): print('already exists')
        else: os.mkdir(f'{save_filepath}/gifs/anat{dof}dof/{sub}')
        for i in range(1,11):
            try:
                img = nib.load(f'{base}/{i}/{sub}.nii.gz')
            except: continue
            plotting.plot_anat(img, cut_coords=(0,0,0), title=f'{i}', dim=-0.5, output_file=f'{save_filepath}/gifs/anat{dof}dof/{sub}/{i}.png')
        frames = [Image.open(f'{save_filepath}/gifs/anat{dof}dof/{sub}/{image}') for image in os.listdir(f'{save_filepath}/gifs/anat{dof}dof/{sub}')]
        frame_one = frames[0]
        
        frame_one.save(f"{save_filepath}/gifs_anat{dof}dof_{sub}.gif", format="GIF", append_images=frames,save_all=True, duration=200, loop=0)
        

def make_gifs_oasis(base, save_filepath):

    dir = os.listdir(f'{base}')[0]
    subs = [x.split('.')[0] for x in os.listdir(f'{base}/{dir}') if 'nii.gz.1.nii.gz' in x]

    # print(subs)
    for sub in subs:
        print(sub)
        # files = glob.glob(f"/home/ine5/projects/rrg-glatard/brainhack-2023-linear-registration/results/mca/*/{sub}.nii.gz")
        if os.path.isdir(f'{save_filepath}/gifs/{sub}'): print('already exists')
        else: os.mkdir(f'{save_filepath}/gifs/{sub}')
        for i in os.listdir(f'{base}'):
            try:
                img = nib.load(f'{base}/{i}/{sub}.nii.gz.1.nii.gz')
            except: continue
            plotting.plot_anat(img, cut_coords=(0,0,0), title=f'{i}', dim=-0.5, output_file=f'{save_filepath}/gifs/{sub}/{i}.png')
        frames = [Image.open(f'{save_filepath}/gifs/{sub}/{image}') for image in os.listdir(f'{save_filepath}/gifs/{sub}')]
        frame_one = frames[0]
        
        frame_one.save(f"{save_filepath}/gifs_{sub}.gif", format="GIF", append_images=frames,save_all=True, duration=200, loop=0)
        # break
        


if __name__ == "__main__":
    
    args = parse_args()
    # make_gifs(args.base, args.dof, args.save_filepath)
    make_gifs_oasis(args.base, args.save_filepath)