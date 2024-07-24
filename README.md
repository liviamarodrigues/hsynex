# H-SynEx

A tool for the hypothalamus and its subregions capable of working across different MRI modalities and resolution. H-SynEx was trained on synthetic images derived from high-resolution *ex vivo* MRI.

The model was trained on synthetic images derived from ex vivo MRI label maps:

<img src=https://github.com/liviamarodrigues/hsynex/blob/main/github1.png>

**Example of testing images:**

<img src=https://github.com/liviamarodrigues/hsynex/blob/main/qualitative_data.png>

**Labels**

| Label | Subregion |
| ------| --------- |
| 1  | left anterior inferior |
| 2  | left posterior|
| 3  | left tuberal inferior |
| 4  | left tuberal superior |
| 6  | left anterior superior |
| 7  | right anterior inferior |
| 8  | right  posterior |
| 9  | right tuberal inferior |
| 10 | right tuberal superior |
| 12 | right anterior superior |

### DEPENDENCIES

This code was implemented using Python 3.10. 

You can install libs using the environment.yml file:

```
conda env create -f environment.yml
```

This repository also relies on a modified implementation of [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet), which is already included.

Finally, it is necessary to install [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) 7.4

## USAGE

To use the tool, please follow the following steps:

- Download this repository
- Using the terminal, enter the inference folder:
  
       cd <your_path>/hsynex/inference
  
- Run the 'find_hypothalamus_subnuclei.py':

        python --input_path <input_path> --out_path: <out_path>

Where:

- `<input_path>` is the path where the input MR images are located
- `<out_path>` is the chosen path to save the segmentations

## CITATION

If you use H-SynEx in your project, please cite:

[[arXiv](https://arxiv.org/pdf/2401.17104.pdf)]: Rodrigues, Livia, et al. **"H-SynEx: Using synthetic images and ultra-high resolution ex vivo MRI for hypothalamus subregion segmentation."** arXiv preprint arXiv:2401.17104 (2024).

