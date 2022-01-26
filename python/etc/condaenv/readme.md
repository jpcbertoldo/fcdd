This will create a conda env for fcdd named `fcdd_rc21`, where 'rc21' stands for "Reproducibility Challenge 2021".

After creating the environment, activate it, go to the fcdd/python and do `pip install --editable .`. 

Or run `fcdd_rc21.sh`.


---


Thalassa

I could not run the code on thalassa because the current CUDA version (10.1) is not enough for pytorch (>=10.2).

Downgrading pytorch

The max pytorch version with CUDA 10.1 is 1.8.1
src: https://pytorch.org/get-started/previous-versions/#v181

I downgraded torch to 1.8.1 and torchvision to 0.9.1 using cu101 (cudatoolkit 10.1).

I tried several solutions, and the one that worked was https://stackoverflow.com/a/66833948/9582881.

When installing fcdd i removed torch and torchvision from the dependencies so they would be installed separately -- that's what the file `requirements-no-torch.txt` is for.

Use `fcdd_rc21_torch181.sh` for thalassa.

Obs: my stackoverflow answer to something similar: https://stackoverflow.com/a/70775341/9582881

