import torch
import torchvision
import torchvision.transforms as transforms

def load_data():
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        crop_scale = 0.08
        jitter_param = 0.4
        lighting_param = 0.1
        image_size = 56
        image_resize = 64

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            # Lighting(lighting_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        valid_transforms = transforms.Compose([
            transforms.Resize(image_resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        train_set = torchvision.datasets.ImageFolder('/data/seed_1993_subset_100_imagenet_500/data/train', transform=train_transforms)
        val_set = torchvision.datasets.ImageFolder('/data/seed_1993_subset_100_imagenet_500/data/val', transform=valid_transforms)
        train_data = torch.utils.data.DataLoader(dataset=train_set, batch_size=256, shuffle=True, drop_last=True, num_workers=8)
        test_data = torch.utils.data.DataLoader(dataset=val_set, batch_size=256, shuffle=False, drop_last=True, num_workers=8)

        return train_data, test_data

def train(model, epoch, train_data, optimizer, criterion):
    # train the local model
    # self.model.to(self.device)
    model.train()
    epoch_loss = []
    batch_loss = []
    for batch_idx, (images, labels) in enumerate(train_data):
        # logging.info(images.shape)
        images, labels = images.to('cuda:0'), labels.to('cuda:0')
        optimizer.zero_grad()
        log_probs = model(images)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    if len(batch_loss) > 0:
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print('(client {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(0,
                                                                    epoch, sum(epoch_loss) / len(epoch_loss)))

def test(model, test_data):
    # self.model.to(self.device)
    model.eval()

    test_correct = 0.0
    test_loss = 0.0
    test_sample_number = 0.0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
            x = x.to('cuda:0')
            target = target.to('cuda:0')

            pred = model(x)
            # loss = self.criterion(pred, target)
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(target).sum()

            test_correct += correct.item()
            # test_loss += loss.item() * target.size(0)
            test_sample_number += target.size(0)
        acc = (test_correct / test_sample_number)*100
        print("************* Server Acc = {:.2f} **************".format(acc))
    return acc

#Load Resnet18
model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(512, 100)
model.to('cuda:0')
criterion = torch.nn.CrossEntropyLoss().to('cuda:0')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)

train_data, test_data = load_data()
for epoch in range(200):
    print('Epoch: {}'.format(epoch))
    train(model, epoch, train_data, optimizer, criterion)
    test(model, test_data)

