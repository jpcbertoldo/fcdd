conda activate fcdd_rc21
cd ${HOME}/fcdd/python/dev
wandb sweep --project unetdd-mvtec-00 --entity mines-paristech-cmm --verbose --name sweep-01 unetdd-mvtec-00-sweep-01.yaml
# sweep id: 8g7n2cc5
# agent line:
# wandb agent mines-paristech-cmm/unetdd-mvtec-00/8g7n2cc5
# full id: mines-paristech-cmm/unetdd-mvtec-00/8g7n2cc5