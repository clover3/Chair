

from data_generator.mask_lm import chunk_lm

data = chunk_lm.DataLoader(256)


train_x = data.get_train_generator()
for i in range(10):
    x,y = train_x.__next__()
    print(x)
    print(y)

