from torchvision import transforms
import os
import pickle

# Strong augmentations
def get_strong_augment(size=384):
    trans = [
        transforms.ToTensor(),
        DivideBy255AndRepeat(),
        transforms.RandomResizedCrop(size=size, scale=(0.25, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR)], p=0.5)
    ]
    train_transform = transforms.Compose(trans)
    return train_transform

# Weak augmentations
def get_weak_augment(size=384):
    trans = [
        transforms.ToTensor(),
        DivideBy255AndRepeat(),
        transforms.RandomResizedCrop(size=size, scale=(0.25, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]

    train_transform = transforms.Compose(trans)
    return train_transform

# Repeat to normalized rgb
class DivideBy255AndRepeat:
    """Transform to scale images to [0, 1] range."""
    def __call__(self, img):
        return (img / 255.).repeat(3, 1, 1)

# Get saved dataviews/dataset images
def get_dataviews():
    save_file_name = "./filtered_data/nrc_dataview_collection_ch-H50_chk-15_p-0.75_w-32.pkl"
    if os.path.exists(os.path.abspath(save_file_name)):
        print(f"Loading: {save_file_name}")
        with open(save_file_name, 'rb') as f:
            return pickle.load(f)
    print("Did not find dataset.")
    exit(1)
