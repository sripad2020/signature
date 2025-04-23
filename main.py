from flask import Flask, render_template, request, jsonify
import os
import uuid
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications import EfficientNetV2B0
from keras.api.layers import GlobalAveragePooling2D, Dense
from keras.api.models import Model
from keras.api.preprocessing import image
from keras.api.applications.efficientnet_v2 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit



def initialize_model():
    # Load pre-trained EfficientNetV2
    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # Load your pre-trained weights (replace with your actual model path)
    # model.load_weights('signature_verification_model.h5')
    return model



model = initialize_model()
def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def compare_signatures(img1_path, img2_path):
    """Compare two signatures and return similarity score"""
    try:
        # Preprocess both images
        img1 = preprocess_image(img1_path)
        img2 = preprocess_image(img2_path)
        embedding_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        emb1 = embedding_model.predict(img1)
        emb2 = embedding_model.predict(img2)
        cosine_similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarity_score = float(cosine_similarity[0][0])
        is_genuine = similarity_score > 0.7

        return {
            'similarity_score': similarity_score,
            'is_genuine': bool(is_genuine),
            'message': "Genuine signature" if is_genuine else "Possible forgery detected"
        }
    except Exception as e:
        return {'error': str(e)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing image files'}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    if image1.filename == '' or image2.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.jpg")
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.jpg")
        image1.save(img1_path)
        image2.save(img2_path)
        result = compare_signatures(img1_path, img2_path)
        result['image1_url'] = img1_path.replace('static/', '')
        result['image2_url'] = img2_path.replace('static/', '')

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        print("Done with similarity comparison")

if __name__ == '__main__':
    app.run(debug=True)