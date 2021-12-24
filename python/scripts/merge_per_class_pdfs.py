from pathlib import Path


here = Path(".")
merge_dir = here / "all_classwise_roc_curves"

merge_dir.mkdir(exist_ok=True)

for dir in here.glob("*"):

    if not dir.is_dir():
        continue 

    target_filepath = dir / "roc.pdf"

    if not target_filepath.exists():
        print(f"{target_filepath=} does not exist")
        continue

    pdf_linkpath = merge_dir / f"{dir.name}.roc.pdf"

    if pdf_linkpath.exists():
        print(f"{pdf_linkpath=} already exists, not replacing!")
        continue

    target_filepath.link_to(pdf_linkpath)

"""
todo
   
done
    fcdd_20211220193242_fmnist_ 
    fcdd_20211220193242_fmnist__HSC
    fcdd_20211220193242_fmnist__AE

    fcdd_20211220193450_fmnist_
    fcdd_20211220193450_fmnist__HSC

    fcdd_20211221161549_cifar10_
"""
