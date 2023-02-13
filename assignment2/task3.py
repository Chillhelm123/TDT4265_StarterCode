import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    use_improved_weight_init = True
    model_plus_weights = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_weights = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_plus_weights, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_weights, val_history_weights = trainer_weights.train(
        num_epochs)

    use_improved_sigmoid = True

    model_plus_sigmoid = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_sigmoid = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_plus_sigmoid, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_sigmoid, val_history_sigmoid = trainer_sigmoid.train(
        num_epochs)
    
    use_momentum = True
    learning_rate = .02

    model_plus_momentum = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_momentum = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_plus_momentum, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_momentum, val_history_momentum = trainer_momentum.train(
        num_epochs)

    use_relu = True


    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Training Loss: Without tricks of the trade", npoints_to_average=10,plot_variance=False)
    utils.plot_loss(
        train_history_weights["loss"], "Training Loss: With Weight Init", npoints_to_average=10,plot_variance=False)
    utils.plot_loss(
        train_history_sigmoid["loss"], "Training Loss: With Weight Init and Improved sigmoid", npoints_to_average=10,plot_variance=False)     
    utils.plot_loss(
        train_history_momentum["loss"], "Training Loss: With Weight Init, Improved sigmoid and momentum", npoints_to_average=10,plot_variance=False) 
    utils.plot_loss(val_history["loss"],
                    "Validation loss: Without tricks of the trade", npoints_to_average=10,plot_variance=False)
    utils.plot_loss(
        val_history_weights["loss"], "Validation loss: With Weight Init", npoints_to_average=10,plot_variance=False)
    utils.plot_loss(
        val_history_sigmoid["loss"], "Validation loss: With Weight Init and Improved sigmoid", npoints_to_average=10,plot_variance=False)     
    utils.plot_loss(
        val_history_momentum["loss"], "Validation loss: With Weight Init, Improved sigmoid and momentum", npoints_to_average=10,plot_variance=False) 
    plt.ylim([0, .4])
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1.05])
    utils.plot_loss(val_history["accuracy"], "Without tricks of the trade")
    utils.plot_loss(
        val_history_weights["accuracy"], "With Weight Init")
    utils.plot_loss(
        val_history_sigmoid["accuracy"], "With Weight Init and Improved sigmoid")
    utils.plot_loss(
        val_history_momentum["accuracy"], "With Weight Init, Improved sigmoid and momentum")
    plt.ylabel("Validation Accuracy")
    plt.savefig("task3_train_loss.png")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
