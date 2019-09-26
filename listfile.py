import pandas as pd
import os

def listfile(folder, split_rate = 0.9):
    train_csv_path = os.path.join(folder, 'Train_label.csv')
    df = pd.read_csv(train_csv_path)
    df['FileName'] = df['FileName'].apply(lambda x: os.path.join(folder,'Train', x))
    df['Code'] = df['Code'].apply(lambda x: x.split(';')[0])
    train_imgs = df['FileName'][:int(len(df)*split_rate)]
    train_labels = df['Code'][:int(len(df)*split_rate)]
    val_imgs = df['FileName'][int(len(df) * split_rate):]
    val_labels = df['Code'][int(len(df) * split_rate):]

    return list(train_imgs),list(train_labels),list(val_imgs),list(val_labels)

if __name__ == '__main__':
    train_imgs, train_labels, val_imgs, val_labels = listfile('./dataset')
    print('number of train set:{}'.format(len(train_imgs)))
    print('number of val set:{}'.format(len(val_imgs)))
    print(val_imgs)
    print(val_labels)