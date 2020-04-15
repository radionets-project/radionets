import pickle
import gzip


def main():
    mnist_path = "./mnist.pkl.gz"

    with gzip.open(mnist_path, "rb") as f:
        ((train_x, train_y), (valid_x, valid_y), _) = pickle.load(f, encoding="latin-1")

    train_x_small = train_x[0:10]
    train_y_small = train_y[0:10]
    valid_x_small = valid_x[0:10]
    valid_y_small = valid_y[0:10]
    test = (train_x_small, train_y_small), (valid_x_small, valid_y_small), ("nothing")

    with gzip.open('./mnist_test.pkl.gz', "wb") as f:
        pickle.dump(test, f)


if __name__ == "__main__":
    main()
