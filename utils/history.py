import yaml
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

f = open('../history.txt', 'r')
losses = f.readlines()
f.close()

train, valid = [], []

for loss in losses:
    loss = yaml.load(loss)
    train = train + loss.get('loss')
    valid = valid + loss.get('val_loss')

print('val_loss_xcep = ', valid)
print('train_loss_xcep = ', train)

fig1 = plt.figure(1)
epoch = range(1, len(train) + 1)
plt.plot(epoch, train, 'C2', marker='X')
plt.plot(epoch, valid, '--', marker='>')
plt.legend(['Training Loss', 'Validation Loss'], loc=1, prop={'size': 18})
plt.xlabel('Epochs', fontsize=20)
plt.ylabel(r'$Loss \; \mathcal{L}$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('../lose_curve.jpg')
plt.show()
