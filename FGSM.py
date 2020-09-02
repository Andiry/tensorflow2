import tensorflow as tf
import matplotlib.pyplot as plt

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]


image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)

image = preprocess(image)
image_probs = pretrained_model.predict(image)

_, image_class, confidence = get_imagenet_label(image_probs)
print('{} : {:.2f}% confidence'.format(image_class, confidence * 100))

loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adv_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t. to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients
    signed_grad = tf.sign(gradient)
    return signed_grad

index = 208
label = tf.one_hot(index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adv_pattern(image, label)

def show_prediction(image, eps):
    image_probs = pretrained_model.predict(image)
    _, image_class, confidence = get_imagenet_label(image_probs)
    print('Epsilon {}: {}, {:.2f}% confidence'.format(eps, image_class, confidence * 100))


epsilons = [0, 0.01, 0.1, 0.15]
for eps in epsilons:
    adv_x = image + eps * perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1) 
    show_prediction(adv_x, eps)
