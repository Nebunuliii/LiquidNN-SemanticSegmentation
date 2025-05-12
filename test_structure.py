from torchinfo import summary

from models.resnet.resnet18 import ResNet18SemanticSegmentation

if __name__ == "__main__":
    model = ResNet18SemanticSegmentation(num_classes=32)
    summary(model, (8, 3, 512, 512), depth=5, col_names=["input_size", "output_size"], verbose=1)
    print()
    print()
    print("--------------------------------------Layers--------------------------------------------")
    print()
    for layer in model.children():
        print(layer)
