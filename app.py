import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

IMG_SIZE = 224

RECOMMENDATIONS = {
    'Potato___Early_blight': [
        'Remove and destroy affected leaves immediately.',
        'Apply a copper-based or chlorothalonil fungicide.',
        'Avoid overhead irrigation; water at the base.',
        'Rotate crops next season to break the disease cycle.',
    ],
    'Potato___Late_blight': [
        'Act immediately — Late Blight spreads rapidly.',
        'Apply a systemic fungicide (e.g., Metalaxyl or Mancozeb).',
        'Remove and bag all infected plant material; do not compost.',
        'Alert neighboring farmers to prevent regional spread.',
    ],
    'Potato___healthy': [
        'Your plant appears healthy!',
        'Continue monitoring weekly for early signs of disease.',
        'Maintain consistent watering and balanced fertilization.',
    ],
}


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('potato_blight_model.keras')
    class_names = np.load('class_names.npy', allow_pickle=True).tolist()
    return model, class_names


def get_gradcam(model, img_array, last_conv_layer='Conv_1'):
    base_model = model.get_layer('mobilenetv2_1.00_224')
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    base_grad_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(last_conv_layer).output, base_model.output]
    )

    gap_layer     = next(l for l in model.layers if 'global_average_pooling' in l.name)
    dropout_layer = next(l for l in model.layers if 'dropout' in l.name)
    dense_layer   = next(l for l in model.layers if 'dense' in l.name)

    preprocessed = preprocess(tf.cast(img_array, tf.float32))

    with tf.GradientTape() as tape:
        conv_outputs, base_out = base_grad_model(preprocessed)
        tape.watch(conv_outputs)
        x           = gap_layer(base_out)
        x           = dropout_layer(x, training=False)
        predictions = dense_layer(x)
        top_class   = tf.argmax(predictions[0])
        loss        = predictions[:, top_class]

    grads   = tape.gradient(loss, conv_outputs)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)
    return heatmap


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = heatmap_colored * 0.4 + original_img
    return np.clip(superimposed / superimposed.max() * 255, 0, 255).astype('uint8')


# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Potato Blight Detector', page_icon='Potato Blight Detector', layout='centered')
st.title('Potato Blight Detector')
st.caption('Upload a photo of a potato leaf to detect Early Blight, Late Blight, or confirm it is Healthy.')

uploaded = st.file_uploader('Upload a leaf image', type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded:
    model, class_names = load_model()

    img = Image.open(uploaded).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_input = np.expand_dims(img_array, 0)

    with st.spinner('Analysing...'):
        preds      = model.predict(img_input, verbose=0)[0]
        pred_idx   = int(np.argmax(preds))
        pred_class = class_names[pred_idx]
        confidence = float(preds[pred_idx])

        heatmap    = get_gradcam(model, img_input)
        overlay    = overlay_heatmap(img_array, heatmap)

    # Results
    label = pred_class.replace('Potato___', '').replace('_', ' ')
    color = 'green' if 'healthy' in pred_class.lower() else 'red'
    st.markdown(f'### Diagnosis: :{color}[{label}]')
    st.progress(confidence, text=f'Confidence: {confidence:.1%}')

    col1, col2 = st.columns(2)
    col1.image(img_array.astype('uint8'), caption='Uploaded Image', use_container_width=True)
    col2.image(overlay, caption='Grad-CAM Heatmap (areas the model focused on)', use_container_width=True)

    st.subheader('📋 Recommended Actions')
    for tip in RECOMMENDATIONS[pred_class]:
        st.markdown(f'- {tip}')

    st.subheader('📊 All Class Probabilities')
    for name, prob in zip(class_names, preds):
        st.progress(float(prob), text=f"{name.replace('Potato___','').replace('_',' ')}: {prob:.1%}")
