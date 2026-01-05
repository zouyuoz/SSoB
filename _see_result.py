from _init import *

def plot_loss(history):
    plt.plot(history['val_loss'], label='val')
    plt.plot(history['train_loss'], label='train')
    plt.title('Loss per epoch'); plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('PLOT/loss.png')
    plt.close()

def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU')
    plt.plot(history['val_miou'], label='val_mIoU')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('PLOT/score.png')
    plt.close()

def plot_acc(history):
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('PLOT/acc.png')
    plt.close()

def plot_lr(history):
    plt.plot(history['lrs'])
    plt.title('Learning Rate per epoch')
    plt.ylabel('lr')
    plt.xlabel('batch')
    plt.grid()
    plt.savefig('PLOT/lr.png')
    plt.close()

ckpt = torch.load('history_checkpoint_B.pth')
# print(ckpt.keys())
history = ckpt['history']
# val_loss = history['last_best_epoch']
# print(history)

plot_loss(history)
plot_score(history)
plot_acc(history)
plot_lr(history)