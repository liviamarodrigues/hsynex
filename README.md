# H-SynEx

A tool for the hypothalamus and its subregions capable of working across different MRI modalities and resolution. H-SynEx was trained on synthetic images derived from high-resolution *ex vivo* MRI.

The model was trained on synthetic images derived from ex vivo MRI label maps:

<img src=https://github.com/liviamarodrigues/hsynex/blob/main/github1.png>

Example of testing images:

<img src=https://github.com/liviamarodrigues/hsynex/blob/main/qualitative_data.png>


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

### DEPENDENCIES

This code was implemented using Python 3.10. To run it, install the dependencies listed at `requirements.txt`.

This repository also relies on a modified implementation of [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet), which is already included.

Finally, it is necessary to install [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) 7.4

## CITATION
