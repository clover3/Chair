import tensorflow_datasets as tfds
# import tfds-nightly as tfds



def main():
    dataset_name = "sci_tail"
    all_dataset = tfds.load(name=dataset_name)
    n = 0
    for e in all_dataset['train']:
        for k, v in e.items():
            print(k, v.numpy())
        print()
        n += 1

        if n > 10:
            break



if __name__ == "__main__":
    main()