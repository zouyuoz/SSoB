from _init import *
from _data import *
from _loss import *
from _UNet import *

# Helper functions saves the states
CHECKPOINT = "checkpoint_C.pth"
HISTORY = "history_checkpoint_C.pth"

def save_checkpoint(epoch, model, optimizer, scheduler, loss, not_improve, path=CHECKPOINT):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'not_improve': not_improve
    }
    torch.save(ckpt, path)
    print(f"💾 Checkpoint saved at epoch {epoch+1} → {path}")

def load_checkpoint(model, optimizer, scheduler, path=CHECKPOINT):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    loss = ckpt['loss'] if 'loss' in ckpt else np.inf
    not_improve = ckpt['not_improve'] if 'not_improve' in ckpt else 0
    print(f"🔄 Checkpoint loaded → resume from epoch {start_epoch}")
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device) # 將內部狀態張量移動到 GPU
    return start_epoch, loss, not_improve

def save_history_checkpoint(e, history_data, path=HISTORY):
    ckpt = {'last_finished_epoch': e, 'history': history_data}
    torch.save(ckpt, path)
    print(f"💾 History saved to: {path}")
    return

def load_history_checkpoint(path=HISTORY):
    if os.path.exists(path):
        ckpt = torch.load(path)
        print(f"🔄 History loaded from: {path}")
        return ckpt['history'] # 返回 history 字典
    else:
        print(f"⚠️ History file not found at {path}, starting fresh.")
        return None

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(
    start_epoch,
    epochs,
    model,
    train_loader,
    val_loader,
    criterion1,
    criterion2,
    optimizer,
    scheduler,
    LL,
    LNI,
    avaliable_training_time,
    accumulation_steps=4,
    resume_history=None
):
    # [修改] 初始化邏輯：如果有 resume_history 就讀取，否則建立空 list
    if resume_history is not None:
        train_losses = resume_history['train_loss']
        test_losses = resume_history['val_loss']
        train_iou = resume_history['train_miou']
        val_iou = resume_history['val_miou']
        train_acc = resume_history['train_acc']
        val_acc = resume_history['val_acc']
        lrs = resume_history['lrs']
        last_best_epoch = resume_history.get('last_best_epoch', 0)
        run_epochs = resume_history.get('run_epochs', 0)
        # 注意：die_out 通常不需延續，它是根據當次執行時間判斷
    else:
        train_losses, test_losses = [], []
        val_iou, val_acc, train_iou, train_acc, lrs = [], [], [], [], []
        last_best_epoch, run_epochs = 0, 0

    min_loss = LL
    decrease, not_improve = 1, LNI
    fit_time = time.time()
    avg_epoch_t = 0.0
    die_out = False
    
    # 確保 history 字典引用的是上述變數
    history = {
        'train_loss': train_losses, 'val_loss': test_losses,
        'train_miou': train_iou, 'val_miou': val_iou,
        'train_acc': train_acc, 'val_acc': val_acc,
        'lrs': lrs, 'last_best_epoch': last_best_epoch,
        'run_epochs': run_epochs, 'die_out': die_out
    }

    model.to(device)

    for e in range(start_epoch, epochs):
        running_loss, iou_score, accuracy = 0.0, 0.0, 0.0
        run_epochs += 1
        since = time.time()

        # training phase
        model.train()
        for i, (image, mask) in enumerate(tqdm(train_loader, bar_format='{l_bar}{bar:60}{r_bar}')):
            image, mask = image.to(device), mask.to(device)

            output = model(image)

            # Check for NaNs in model output
            if torch.isnan(output).any(): print(f"NaN found in model output @ batch {i}")
            loss1 = criterion1(output, mask) 
            loss2 = criterion2(output, mask)
            loss = (loss1 + loss2) / accumulation_steps

            loss.backward()
            running_loss += loss.item() * accumulation_steps
            iou_score += mIoU(output, mask, n_classes=3)
            accuracy += pixel_accuracy(output, mask)

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            lrs.append(get_lr(optimizer))

        # evalue phase
        model.eval()
        test_loss, test_accuracy, val_iou_score = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, (image, mask) in enumerate(tqdm(val_loader, bar_format='{l_bar}{bar:60}{r_bar}')):
                image, mask = image.to(device), mask.to(device)

                output = model(image)
                loss1 = criterion1(output, mask)
                loss2 = criterion2(output, mask)
                loss = loss1 + loss2

                val_iou_score += mIoU(output, mask, n_classes=3)
                test_accuracy += pixel_accuracy(output, mask)
                test_loss += loss.item()

        # calculate mean for each metrics
        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(val_loader))
        train_acc.append((accuracy/len(train_loader)).cpu().item())
        val_acc.append((test_accuracy/len(val_loader)).cpu().item())
        train_iou.append(iou_score/len(train_loader))
        val_iou.append(val_iou_score/len(val_loader))

        # update average epoch time
        this_epoch_t = time.time()-since
        avg_epoch_t = (avg_epoch_t + this_epoch_t/60) / 2 if avg_epoch_t else this_epoch_t/60
        die_out = avaliable_training_time - (time.time()-fit_time)/60 < avg_epoch_t

        if min_loss >= (test_loss/len(val_loader)):
            print('👏 Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(val_loader))))
            min_loss = (test_loss/len(val_loader))
            not_improve = 0
            last_best_epoch = e
            torch.save(model.state_dict(), "BEST.pt")
        else:
            if abs(min_loss - test_loss/len(val_loader)) > 0.002:
              not_improve += 1
              print(f'👮 Loss Not exceed best for {not_improve} time')
            # if not_improve == 10:
            #     print(f'Loss not decrease for {not_improve} times, Stop Training')
            #     break

        print(
            f"🏷️  Epoch[{e+1}/{epochs}] "
            f"Train Loss: {running_loss/len(train_loader):.3f}| "
            f"Val Loss: {test_loss/len(val_loader):.3f}| "
            f"Train mIoU: {iou_score/len(train_loader):.3f}| "
            f"Val mIoU: {val_iou_score/len(val_loader):.3f}| "
            f"Time: {datetime.timedelta(seconds=int(this_epoch_t))}"
        )
        history['last_best_epoch'] = last_best_epoch
        history['run_epochs'] = run_epochs

        save_checkpoint(e, model, optimizer, scheduler, test_loss/len(val_loader), not_improve)
        save_history_checkpoint(e, history)
        if die_out:
            print("No time to train, Save the status and end"); break

    print('Total time: ', datetime.timedelta(seconds=int(time.time() - fit_time)))
    return history

# ---
# model = UNet(n_channels=3, n_classes=3)

# or
from _AttnUNet import AttentionUNet
model = AttentionUNet(in_channels=3, out_channels=3)

max_lr = 5e-3
start_epoch = 0
epoch = 100
weight_decay = 1e-4

# criterion1 = nn.CrossEntropyLoss()
# criterion1 = FocalLoss(alpha=[0.3502, 0.2864, 0.3634], gamma=2.0)
criterion1 = FocalLoss()
criterion2 = DiceLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader),
    div_factor=10.0, final_div_factor=1000.0)
ATT = 9999

# --- real train ---
start_epoch = 0
last_loss = 1.00
last_not_improve = 0
use_checkpoint = CHECKPOINT
resume_history_data = None
start_epoch, last_loss, last_not_improve = load_checkpoint(model, optimizer, scheduler, path=use_checkpoint)
resume_history_data = load_history_checkpoint(path=HISTORY)

torch.autograd.set_detect_anomaly(True)
history = fit(
    start_epoch, epoch, model, train_loader, val_loader,
    criterion1, criterion2, optimizer, scheduler,
    last_loss, last_not_improve, ATT, accumulation_steps=4, resume_history=resume_history_data
)