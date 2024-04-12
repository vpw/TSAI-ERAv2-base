from main import *

# get CL args
# parse_arguments()
# get epochs, lr, lr scheduler, optimizer, criterion?


# device - cuda or not
get_device()

# data transformers
transform_train, transform_test = set_transforms()

# data split train and test
trainset, trainloader, testset, testloader, classes = prepare_data()

# get model
# pass the model type
global net
net = get_model()
get_model()
print(net)

# net to device
net = net_to_device()
net_to_device()

# criterion, optimizer, scheduler
criterion, optimizer, scheduler = c_o_s(net)

# training and test loops


# epochs

# batch size

# choose optimizer

# scheduler to use

#for epoch in range(start_epoch, start_epoch+200):
for epoch in range(0,20):
    train(epoch, trainloader, optimizer, criterion)
    test(epoch, testloader, criterion)
    scheduler.step()
