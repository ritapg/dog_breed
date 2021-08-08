import numpy as np
import torch

# train model function


def train(n_epochs, trainloader, validloader, model, optimizer, criterion, use_cuda, save_path):
        """returns trained model"""
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf

        for epoch in range(1, n_epochs + 1):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            # train the model #
            model.train()
            # for data, target in train_loader:
            for batch_idx, (data, target) in enumerate(trainloader):

                    if use_cuda:
                        data, target = data.cuda(), target.cuda()

                    optimizer.zero_grad()
                    # forward pass
                    output = model(data)
                    # batch loss
                    loss = criterion(output, target)
                    # backward pass
                    loss.backward()
                    # optimization step (parameter update)
                    optimizer.step()
                    # update training loss
                    train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # validate the model
            model.eval()
            for batch_idx, (data, target) in enumerate(validloader):

                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## update the average validation loss

                # forward pass
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
            ))

            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model, 'models/model_.pt')
                #torch.save(model.state_dict(), save_path)
                valid_loss_min = valid_loss

        # return trained model
        return model


# test function
def test(testloader_transf, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(testloader_transf):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        ## forward pass
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
