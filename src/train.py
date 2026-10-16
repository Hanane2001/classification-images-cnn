import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing import main_preprocessing


def creer_model_cnn(kr_size=(3, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(32, kernel_size=kr_size, strides=(1, 1), padding='valid', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, kernel_size=kr_size, strides=(1, 1), padding='valid', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    return model

def creer_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.2, min_delta=0.001, min_lr=1e-6, mode='auto')
    checkpoint = tf.keras.callbacks.ModelCheckpoint("../models/final_model.keras", monitor="val_loss", save_best_only=True)
    return [early_stopping, lr_reduce, checkpoint]

def compiler_model(model):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def entrainer_model(model, train_ger, val_ger):
    callbacks = creer_callbacks()
    history = model.fit(train_ger, epochs=30, validation_data=val_ger, callbacks=callbacks, verbose=2)
    return history

def evaluate_model(model, test_ds):
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)
    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {test_accuracy}')
    return test_loss, test_accuracy

def visualiser_historique(history):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

def sauvegarder_model(model):
    model.save("../models/final_model.keras")
    print("Modele sauvegarde avec succes")

def main_train():
    train_ds, validation_ds, test_ds, class_to_index = main_preprocessing()
    model = creer_model_cnn()
    model = compiler_model(model)
    model.summary()
    history = entrainer_model(model, train_ds, validation_ds)
    evaluate_model(model, test_ds)
    visualiser_historique(history)
    sauvegarder_model(model)

if __name__ == "__main__":
    main_train()