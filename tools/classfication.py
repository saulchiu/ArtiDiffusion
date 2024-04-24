import timm
import detectors

net = model = timm.create_model("resnet18_cifar10", pretrained=True)
print()