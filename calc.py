
import os
import significantdigits as sd
# import torchio as tio
import nibabel as nib
from significantdigits import Error, Method
import numpy as np
import glob


def calculate_fuzzy(folder, save_filepath):

    t1s = []

    files = os.listdir(f"{folder}")

    for subject in os.listdir(f"{folder}/1"):
        if 'mat' in subject:continue
        if subject.split('.')[0] in already_done: continue

        print(subject)

        if 'OAS1_0041_MR1_mpr-3_anon_reorient.nii.gz' != subject: continue

        for iter in range(1,11):
        
            if 'nii' in subject: 
                # img = tio.ScalarImage(f"{folder}/{iter}/{subject}")
                img = nib.load(f"{folder}/{iter}/{subject}")
                print(type(img.get_fdata().squeeze()[0,0,0]))
                print(img.get_fdata().shape)
                t1s.append(img.get_fdata().squeeze())

        mean = np.mean(np.array(t1s), axis=0)
        print(mean.shape, type(mean[0,0,0]))


        sig = sd.significant_digits(
                array=t1s,
                reference=mean,
                # axis=0,
                # error=Error.Relative,
                # method=Method.General,
            )
        
        print(type(sig[0,0,0]), sig.shape)

        
        np.save(f"{save_filepath}/{subject.split('.')[0]}.npy", sig)
        # break

def calculate_archdocker(save_filepath):

    t1s = []

    for subject in glob.glob(f"g5k_results/Docker/6a92985a9f458557cd62fb3eb0f0cebf/*1.nii.gz"):

        print(subject)
        if os.path.isfile(f"g5k_results/sigmaps/Docker/{subject.split('/')[-1].split('.')[0]}.npy"): continue
        # subject = subject.split('/')[-1].split('.')[0]
        # files[subject] = []

        for dir in glob.glob(f"g5k_results/Docker/*"):
            t1s.append(nib.load(f"{subject}").get_fdata().squeeze())
            # files[subject].append(f"{dir}/{subject}")

        # for dir in glob.glob(f"g5k_results/GUIX/*"):
        #     # files[subject].append(f"{dir}/{subject}")
        #     t1s.append(nib.load(f"{subject}").get_fdata().squeeze())
        
        mean = np.mean(np.array(t1s), axis=0)
        print(mean.shape, type(mean[0,0,0]))


        sig = sd.significant_digits(
                array=t1s,
                reference=mean,
            )
        
        print(type(sig[0,0,0]), sig.shape)

        
        np.save(f"{save_filepath}/{subject.split('/')[-1].split('.')[0]}.npy", sig)


def calculate_archguix(save_filepath):

    t1s = []

    for subject in glob.glob(f"g5k_results/Docker/6a92985a9f458557cd62fb3eb0f0cebf/*1.nii.gz"):

        print(subject)
        if os.path.isfile(f"g5k_results/sigmaps/GUIX/{subject.split('/')[-1].split('.')[0]}.npy"): continue
        # subject = subject.split('/')[-1].split('.')[0]
        # files[subject] = []

        # for dir in glob.glob(f"g5k_results/Docker/*"):
        #     t1s.append(nib.load(f"{subject}").get_fdata().squeeze())
        #     # files[subject].append(f"{dir}/{subject}")

        for dir in glob.glob(f"g5k_results/GUIX/*"):
            # files[subject].append(f"{dir}/{subject}")
            t1s.append(nib.load(f"{subject}").get_fdata().squeeze())
        
        mean = np.mean(np.array(t1s), axis=0)
        print(mean.shape, type(mean[0,0,0]))


        sig = sd.significant_digits(
                array=t1s,
                reference=mean,
            )
        
        print(type(sig[0,0,0]), sig.shape)

        
        np.save(f"{save_filepath}/{subject.split('/')[-1].split('.')[0]}.npy", sig)


def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


def calculate_swarm():

    fuzzy = []
    arch = []
    for sub in os.listdir('results/sigmaps'):
        print(sub)
        s = np.load(f"results/sigmaps/{sub}").flatten()
        fuzzy.append(s)
    fuzzy = np.array(fuzzy).flatten()
    print(fuzzy.shape)

    for sub in os.listdir('g5k_results/sigmaps'):
        print(sub)
        s = np.load(f"g5k_results/sigmaps/{sub}").flatten()
        arch.append(s)
    arch = np.array(arch).flatten()
    print(arch.shape)


    x1 = simple_beeswarm(fuzzy)
    # ax.plot(x, prot['Verrou All'], 'o', alpha=0.5)
    x2 = simple_beeswarm(arch)

    np.save('swarm_fuzzy.pkl', x1)
    np.save('sig_fuzzy.pkl', fuzzy)
    np.save('swarm_arch.pkl', x2)
    np.save('sig_arch.pkl', arch)




# calculate_swarm()
calculate_archdocker(save_filepath='g5k_results/sigmaps/Docker')
calculate_archguix(save_filepath='g5k_results/sigmaps/GUIX')
# calculate_fuzzy(folder='results/anat-12dofs/mca', save_filepath='results/sigmaps')
