# SanDiffusion


gtsrb
```shell
python sandiffusion.py --config-name gtsrb attack=badnet
```

if using sigmoid schedule, then run this:
```shell
python sandiffusion.py --config-name gtsrb attack=badnet diffusion.beta_schedule=sigmoid p_start=200 p_end=400
```

# about the partial step

when using linear beta schedule, the partial step is 100~200. When we diffuse the poisoniong sample 300 steps, the badnet trigger will be nearly completely destroyed.

when using sigmoid schedule, the partical step is 200~400

And there is no need to set the p_end to 1000(cause the trigger is absolutely destroyed, the Unet will learn nothing but distort the normal decision boundary). What's more, the Unet will not link the trigger with backdoor.




