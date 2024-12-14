# app.py

import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify, session, flash
from flask_restful import Api, Resource # type: ignore
from werkzeug.utils import secure_filename
from flask_cors import CORS # type: ignore
import uuid
from flask_session import Session # type: ignore
import base64
import hashlib
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from forms import RegistrationForm, LoginForm
from flask_bcrypt import Bcrypt # type: ignore
from models import db, User, Image  # Import from models.py

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure key in production
CORS(app)
api = Api(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
PARENT_DIR = 'RSSCN7-master'
FEATURES_DIR = 'features'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Flask-Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'
app.config['SESSION_PERMANENT'] = False
Session(app)

# Ensure necessary directories exist
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# Initialize SQLAlchemy with the app
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # Database URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect to 'login' for @login_required
login_manager.login_message_category = 'info'

# Initialize Bcrypt
bcrypt = Bcrypt(app)

# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Configuration
UPLOAD_FOLDER = 'uploads'  # Not used in transformations anymore
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
PARENT_DIR = 'RSSCN7-master'  # Existing image dataset
FEATURES_DIR = 'features'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Flask-Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'  # Directory to store session files
app.config['SESSION_PERMANENT'] = False
Session(app)  # Initialize Flask-Session

# Ensure necessary directories exist
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# In-memory storage for image data
image_data = []  # List of dictionaries with 'id', 'path', and 'features'

# Separate NumPy array for features to enable vectorized operations
features_matrix = None  # Will be initialized after loading images

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- Feature Extraction Functions -----------------

# Function to generate a unique feature file name based on image path
def get_feature_filename(image_path):
    # Use a hash of the image path to ensure unique and filesystem-friendly filenames
    hash_digest = hashlib.md5(image_path.encode('utf-8')).hexdigest()
    return os.path.join(FEATURES_DIR, f"{hash_digest}.npz")

# Helper function to convert RGB to HEX
def rgb_to_hex(r, g, b):
    """Convert RGB values to HEX format."""
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# Function to extract color histograms from images
def extract_histogram(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)  # Normalize, flatten, and convert to float32
    return hist

# Function to extract dominant colors using K-Means
def extract_dominant_colors(image_path, k=3):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k, random_state=42)
    labels = clt.fit_predict(image)
    dominant_colors = clt.cluster_centers_.flatten().astype(np.float32)
    print(f"Extracted Dominant Colors (RGB): {dominant_colors}")  # Debugging statement
    return dominant_colors

# New Function: Extract Dominant Colors Without Normalization
def get_dominant_colors_original(image_path, k=3):
    """
    Extract dominant colors from an image without normalization.

    Parameters:
        image_path (str): Path to the image file.
        k (int): Number of dominant colors to extract.

    Returns:
        np.ndarray: Array of dominant colors in RGB format as integers.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k, random_state=42)
    labels = clt.fit_predict(image)
    dominant_colors = clt.cluster_centers_.astype(int)  # Convert to integer RGB values
    return dominant_colors

# Function to extract Gabor texture features with summary statistics
def extract_gabor_features(image_path, frequencies=(0.1, 0.3, 0.5)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    gabor_features = []
    for frequency in frequencies:
        kernel = cv2.getGaborKernel((21, 21), 8.0, 0, frequency, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        # Compute summary statistics instead of flattening
        gabor_features.append(filtered.mean())
        gabor_features.append(filtered.var())
        gabor_features.append(filtered.max())
        gabor_features.append(filtered.min())
    return np.array(gabor_features, dtype=np.float32)

# Function to extract Hu Moments
def extract_hu_moments(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log transform to bring values to a similar scale
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments.astype(np.float32)

# Function to extract Local Binary Patterns (LBP)
def extract_lbp(image_path, P=24, R=3):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    lbp = local_binary_pattern(image, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist

# Function to extract SIFT features and compute descriptors
def extract_sift_features(image_path, num_features=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    sift = cv2.SIFT_create(nfeatures=num_features)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is not None:
        # Use mean of descriptors
        descriptors_mean = descriptors.mean(axis=0)
    else:
        # If no descriptors are found, return a zero vector
        descriptors_mean = np.zeros(128, dtype=np.float32)
    return descriptors_mean

# Function to compute the combined feature vector
def extract_features(image_path):
    hist = extract_histogram(image_path)
    dominant_colors = extract_dominant_colors(image_path)
    gabor = extract_gabor_features(image_path)
    hu = extract_hu_moments(image_path)
    lbp = extract_lbp(image_path)
    sift = extract_sift_features(image_path)
    # Concatenate all features into a single vector
    feature_vector = np.concatenate([hist, dominant_colors, gabor, hu, lbp, sift]).astype(np.float32)
    # Normalize the feature vector
    feature_vector = cv2.normalize(feature_vector, None).flatten()
    return feature_vector

# Function to load a feature vector from a .npz file
def load_feature(image_path):
    feature_file = get_feature_filename(image_path)
    if not os.path.exists(feature_file):
        # Feature file does not exist, extract and save
        try:
            features = extract_features(image_path)
            np.savez_compressed(feature_file, features=features)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    # Load the feature vector
    data = np.load(feature_file)
    return data['features']

# Function to compute Euclidean distance between two feature vectors
def compute_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# ----------------- Initialize Image Database -----------------

def init_image_database():
    global image_data, features_matrix
    print("Loading the images database...")
    image_paths = []
    # Read all images from the PARENT_DIR
    dir_files = os.listdir(PARENT_DIR)
    for i, _dir in enumerate(dir_files):
        if _dir == 'Query':
            continue  # Skip the 'Query' category
        dir_path = os.path.join(PARENT_DIR, _dir)
        if not os.path.isdir(dir_path):
            continue
        print(f"Retrieving from the {i + 1} dir named '{_dir}' ...", end="")
        images_in_dir = [os.path.join(_dir, f).replace("\\", "/") for f in os.listdir(dir_path) if
                         f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        image_paths.extend(images_in_dir)
        print("done!")
    print(f"Retrieved {len(image_paths)} images.")

    # Extract features for all images
    print("Extracting features for all images...")
    for idx, rel_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        image_id = str(uuid.uuid4())
        image_data.append({'id': image_id, 'path': rel_path, 'features': None})  # Placeholder
        # Extract and store features
        full_path = os.path.join(app.root_path, PARENT_DIR, rel_path)
        try:
            features = load_feature(full_path)
            if features is not None:
                image_data[-1]['features'] = features
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")
            image_data.pop()  # Remove entry if failed
    print(f"Feature extraction completed. Total images with features: {len(image_data)}")

    # Create the features matrix for vectorized operations
    if image_data:
        features_matrix = np.vstack([img['features'] for img in image_data if img['features'] is not None]).astype(np.float32)
    else:
        features_matrix = np.array([], dtype=np.float32).reshape(0, 694)  # Adjusted to new feature size

# ----------------- REST API Resources -----------------

class DescriptorAPI(Resource):
    def post(self):
        data = request.json
        image_path = data.get('image_path')
        if not image_path:
            return {'error': 'Image path not provided'}, 400
        full_path = os.path.join(app.root_path, image_path)
        if not os.path.exists(full_path):
            return {'error': 'Invalid image path'}, 400
        try:
            hist = extract_histogram(full_path)
        except Exception as e:
            return {'error': str(e)}, 500
        return {'histogram': hist.tolist()}

api.add_resource(DescriptorAPI, '/api/descriptor')

# ----------------- Home Route -----------------

@app.route('/')
@login_required
def index():
    categories = [c for c in os.listdir(PARENT_DIR) if os.path.isdir(os.path.join(PARENT_DIR, c))]
    category_images = {}
    for category in categories:
        category_path = os.path.join(PARENT_DIR, category)
        images = [img for img in image_data if img['path'].startswith(category + '/')]
        if images:
            first_image = images[0]['path']
            category_images[category] = first_image
        else:
            category_images[category] = None  # Placeholder if no images
    return render_template('index.html', categories=categories, category_images=category_images, parent_dir=PARENT_DIR)

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        # Hash the password
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        # Create a new user instance
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        # Add to the database
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        # Check if user exists
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            # Log the user in
            login_user(user, remember=form.remember.data)
            flash('You have been logged in!', 'success')
            # Redirect to next page if exists
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check email and password.', 'danger')
    return render_template('login.html', title='Login', form=form)

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ----------------- Category Route -----------------

@app.route('/category/<string:category_name>')
@login_required
def view_category(category_name):
    if category_name == "Query":
        # Retrieve only query images uploaded by the current user
        query_images = Image.query.filter_by(user_id=current_user.id).filter(
            Image.path.like(f"{category_name}/%")
        ).all()
        images = [{'filename': os.path.basename(img.path), 'id': img.id, 'path': img.path} for img in query_images]
    else:
        # Retrieve base dataset images
        base_images = [img for img in image_data if img['path'].startswith(f"{category_name}/")]
        base_images_formatted = [{'filename': os.path.basename(b['path']), 'id': b['id'], 'path': b['path']} for b in base_images]

        # Retrieve user-uploaded images
        user_uploaded_images = Image.query.filter(
            Image.path.like(f"{category_name}/%")
        ).all()
        user_images_formatted = [{'filename': os.path.basename(img.path), 'id': img.id, 'path': img.path} for img in user_uploaded_images]

        # Combine both lists
        images = base_images_formatted + user_images_formatted

    if not images:
        flash('No images found in this category.', 'info')
    return render_template('category.html', category=category_name, images=images)

# ----------------- Upload Route -----------------

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No files part in the request.', 'danger')
            return redirect(request.url)

        files = request.files.getlist('files[]')
        category = request.form.get('category') or 'Uncategorized'
        category_path = os.path.join(PARENT_DIR, secure_filename(category))
        os.makedirs(category_path, exist_ok=True)

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                save_rel_path = os.path.join(category, unique_filename).replace("\\", "/")
                final_full_path = os.path.join(app.root_path, PARENT_DIR, save_rel_path)
                file.save(final_full_path)

                # Extract features and save to database
                try:
                    features = extract_features(final_full_path)
                    uploaded_image = Image(
                        filename=unique_filename,
                        path=save_rel_path,
                        user_id=current_user.id,
                        features=features
                    )
                    db.session.add(uploaded_image)
                    db.session.commit()
                except Exception as e:
                    print(f"Error processing uploaded image {save_rel_path}: {e}")
                    flash(f"Error processing {filename}.", 'danger')

        flash('Images uploaded successfully!', 'success')
        return redirect(url_for('index'))

    # Render upload form
    categories = [c for c in os.listdir(PARENT_DIR) if os.path.isdir(os.path.join(PARENT_DIR, c))]
    return render_template('upload.html', categories=categories)

# ----------------- Delete Route -----------------

@app.route('/delete/<string:image_id>', methods=['POST'])
@login_required
def delete_image(image_id):
    # Find the image by ID in the database
    image = Image.query.get(image_id)
    if not image:
        flash('You do not have permission to delete this image.', 'danger')
        return redirect(url_for('index'))

    # Check ownership
    if image.user_id != current_user.id:
        flash('You do not have permission to delete this image.', 'danger')
        return redirect(url_for('index'))

    # Remove the image file
    full_path = os.path.join(app.root_path, PARENT_DIR, image.path)
    if os.path.exists(full_path):
        try:
            os.remove(full_path)
            print(f"Deleted image file: {full_path}")
        except OSError as e:
            print(f"Error deleting image file {full_path}: {e}")
    else:
        print(f"Image file not found for deletion: {full_path}")

    # Remove the feature file if it exists
    feature_file = get_feature_filename(image.path)
    if os.path.exists(feature_file):
        try:
            os.remove(feature_file)
            print(f"Deleted feature file: {feature_file}")
        except OSError as e:
            print(f"Error deleting feature file {feature_file}: {e}")

    # Remove from in-memory storage
    global image_data, features_matrix
    img_entry = next((img for img in image_data if img['id'] == image_id), None)
    if img_entry:
        idx_to_remove = image_data.index(img_entry)
        image_data.remove(img_entry)
        print(f"Removed image from in-memory storage: ID={image_id}")
        if features_matrix is not None and features_matrix.shape[0] > idx_to_remove:
            features_matrix = np.delete(features_matrix, idx_to_remove, axis=0)

    # Delete the image record from the database
    try:
        db.session.delete(image)
        db.session.commit()
        flash('Image has been deleted!', 'success')
    except Exception as e:
        flash(f"Error deleting image record from the database: {e}", 'danger')
        print(f"Error deleting image record from the database: {e}")

    return redirect(url_for('index'))

# ----------------- Transform Route -----------------

@app.route('/transform/<string:image_id>', methods=['GET', 'POST'])
@login_required
def transform_image(image_id):
    global image_data, features_matrix
    img_entry = next((img for img in image_data if img['id'] == image_id), None)
    if img_entry is None:
        return "Invalid Image ID", 400

    if request.method == 'GET':
        # Just display the page with the current image
        return render_template('transform.html', image_id=image_id, image_path=img_entry['path'])

    # If POST, handle the transformation data
    if request.is_json:
        data = request.get_json()
        transformed_image_data = data.get('transformation_data')
        if transformed_image_data:
            try:
                # Decode the Base64 image
                header, encoded = transformed_image_data.split(',', 1)
                file_ext = header.split('/')[1].split(';')[0]  # e.g., 'png', 'jpeg'
                decoded = base64.b64decode(encoded)
                # Determine the full path of the original image
                original_full_path = os.path.join(app.root_path, PARENT_DIR, img_entry['path'])
                if not os.path.exists(original_full_path):
                    return jsonify({"error": "Original image file not found on server."}), 404

                # Overwrite the original image with the transformed image
                with open(original_full_path, 'wb') as f:
                    f.write(decoded)
                print(f"Transformed image saved to: {original_full_path}")

                # Re-extract features for the updated image
                features = extract_features(original_full_path)
                if features is not None:
                    img_entry['features'] = features
                    idx = image_data.index(img_entry)
                    if features_matrix is not None and idx < features_matrix.shape[0]:
                        features_matrix[idx, :] = features
                    else:
                        # If features_matrix is None or index out of range, append
                        if features_matrix is None or features_matrix.size == 0:
                            features_matrix = features.reshape(1, -1).astype(np.float32)
                        else:
                            features_matrix = np.vstack([features_matrix, features.reshape(1, -1)])
                    
                    return jsonify({"new_path": img_entry['path']})
                else:
                    return jsonify({"error": "No features extracted"}), 500
            except Exception as e:
                print(f"Error processing transformed image: {e}")
                return jsonify({"error": "Failed to process transformed image."}), 500
        else:
            return jsonify({"error": "No transformation data provided"}), 400
    else:
        # If form submission was used instead of AJAX (optional)
        transformed_image_data = request.form.get('transformation_data')
        if transformed_image_data:
            try:
                # Decode the Base64 image
                header, encoded = transformed_image_data.split(',', 1)
                file_ext = header.split('/')[1].split(';')[0]
                decoded = base64.b64decode(encoded)
                # Determine the full path of the original image
                original_full_path = os.path.join(app.root_path, PARENT_DIR, img_entry['path'])
                if not os.path.exists(original_full_path):
                    return "Original image file not found on server.", 404

                # Overwrite the original image with the transformed image
                with open(original_full_path, 'wb') as f:
                    f.write(decoded)
                print(f"Transformed image saved to: {original_full_path}")

                # Re-extract features for the updated image
                features = extract_features(original_full_path)
                if features is not None:
                    img_entry['features'] = features
                    idx = image_data.index(img_entry)
                    if features_matrix is not None and idx < features_matrix.shape[0]:
                        features_matrix[idx, :] = features
                    else:
                        # If features_matrix is None or index out of range, append
                        if features_matrix is None or features_matrix.size == 0:
                            features_matrix = features.reshape(1, -1).astype(np.float32)
                        else:
                            features_matrix = np.vstack([features_matrix, features.reshape(1, -1)])
                    
                    return redirect(url_for('index'))
                else:
                    return "No features extracted", 500
            except Exception as e:
                print(f"Error processing transformed image: {e}")
                return "Failed to process transformed image.", 500
        return redirect(url_for('index'))

# ----------------- Search Route -----------------

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search_image():
    if request.method == 'POST':
        # Clear previous session data
        session.pop('query_image_id', None)
        session.pop('current_iteration', None)
        session.pop('relevant_ids', None)
        session.pop('irrelevant_ids', None)
        session.pop('retrieved_ids', None)

        # Handle search query
        if 'query_image' not in request.files:
            flash('No query image provided.', 'danger')
            return redirect(request.url)

        file = request.files['query_image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            query_filename = f"query_{uuid.uuid4().hex}_{filename}"
            # Save the query image in the 'Query' category within RSSCN7-master
            query_category = 'Query'
            query_category_path = os.path.join(PARENT_DIR, query_category)
            os.makedirs(query_category_path, exist_ok=True)
            save_rel_path = os.path.join(query_category, query_filename).replace("\\", "/")
            final_full_path = os.path.join(app.root_path, PARENT_DIR, save_rel_path)
            file.save(final_full_path)

            # Compute features for the query image
            try:
                query_features = extract_features(final_full_path)
            except Exception as e:
                print(f"Error processing query image {save_rel_path}: {e}")
                flash('Failed to process query image.', 'danger')
                return redirect(request.url)

            # Add query image to database
            query_image = Image(
                filename=query_filename,
                path=save_rel_path,
                user_id=current_user.id,  # Associate with the logged-in user
                features=query_features
            )
            db.session.add(query_image)
            db.session.commit()

            # Add query image to image_data
            image_data.append({'id': query_image.id, 'path': save_rel_path, 'features': query_features})

            global features_matrix
            query_feature = query_features.reshape(1, -1).astype(np.float32)
            if features_matrix is None or features_matrix.size == 0:
                features_matrix = query_feature
            else:
                features_matrix = np.vstack([features_matrix, query_feature])

            # Store iteration limit and final results count set by the user
            iteration_limit = request.form.get('iteration_limit', type=int)
            if iteration_limit is None:
                iteration_limit = 10  # default fallback
            session['iteration_limit'] = iteration_limit

            final_results_count = request.form.get('final_results_count', type=int)
            if final_results_count is None:
                final_results_count = 10  # default fallback
            session['final_results_count'] = final_results_count

            session['query_image_id'] = query_image.id
            session['current_iteration'] = 1
            session['relevant_ids'] = []
            session['irrelevant_ids'] = []

            # Initial retrieval: top N closest images excluding 'Query' category
            valid_indices = [i for i, img in enumerate(image_data) if not img['path'].startswith('Query/')]
            if not valid_indices:
                retrieved_images = []
            else:
                other_features = features_matrix[valid_indices]
                distances = np.linalg.norm(other_features - query_features, axis=1)
                top_indices_within_valid = np.argsort(distances)[:final_results_count]
                top5_indices = [valid_indices[i] for i in top_indices_within_valid]
                retrieved_images = [{'path': image_data[i]['path'], 'id': image_data[i]['id']} for i in top5_indices]

            session['retrieved_ids'] = [img['id'] for img in retrieved_images]
            flash('Search initiated successfully!', 'success')
            return render_template('search.html', query_image=save_rel_path, retrieved_images=retrieved_images, iteration=1)
    else:
        # If GET request, render the search page without images
        return render_template('search.html')

@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    data = request.get_json()
    selected = data.get('selected')  # List of image IDs
    iteration = session.get('current_iteration', 1)
    query_image_id = session.get('query_image_id', None)
    iteration_limit = session.get('iteration_limit', 10)  # default fallback if not set
    final_results_count = session.get('final_results_count', 10)  # default fallback if not set

    if not selected:
        return jsonify({'error': 'No feedback provided'}), 400

    # Validate selected IDs
    selected_ids = []
    for img_id in selected:
        img_entry = next((img for img in image_data if img['id'] == img_id), None)
        if img_entry is None:
            return jsonify({'error': f'Invalid image ID selected: {img_id}'}), 400
        selected_ids.append(img_entry['id'])

    if 'relevant_ids' not in session:
        session['relevant_ids'] = []
    if 'irrelevant_ids' not in session:
        session['irrelevant_ids'] = []

    # Update relevant and irrelevant IDs
    for img_id in selected_ids:
        if img_id in session['relevant_ids']:
            if img_id not in session['irrelevant_ids']:
                session['irrelevant_ids'].append(img_id)
        else:
            session['relevant_ids'].append(img_id)

    if not query_image_id:
        return jsonify({'error': 'No query image found in session'}), 400

    query_image = next((img for img in image_data if img['id'] == query_image_id), None)
    if not query_image:
        return jsonify({'error': 'Query image not found'}), 400

    query_vector = query_image['features']
    # Compute centroids for relevant and irrelevant sets
    if session['relevant_ids']:
        relevant_features = np.array([img['features'] for img in image_data if img['id'] in session['relevant_ids']])
        relevant_centroid = relevant_features.mean(axis=0)
    else:
        relevant_centroid = query_vector

    if session['irrelevant_ids']:
        irrelevant_features = np.array([img['features'] for img in image_data if img['id'] in session['irrelevant_ids']])
        irrelevant_centroid = irrelevant_features.mean(axis=0)
    else:
        irrelevant_centroid = query_vector

    # Rocchio update
    alpha = 1.0
    beta = 0.75
    gamma = 0.15
    updated_query = alpha * query_vector
    if session['relevant_ids']:
        updated_query += (beta / len(session['relevant_ids'])) * relevant_centroid
    if session['irrelevant_ids']:
        updated_query -= (gamma / len(session['irrelevant_ids'])) * irrelevant_centroid
    updated_query = cv2.normalize(updated_query, None).flatten()

    global features_matrix
    excluded_ids = set([query_image_id])

    # If not final iteration, exclude previously selected and retrieved images
    if iteration + 1 <= iteration_limit:
        excluded_ids.update(session.get('relevant_ids', []))
        excluded_ids.update(session.get('irrelevant_ids', []))
        excluded_ids.update(session.get('retrieved_ids', []))
        # Additionally exclude all images from 'Query' category
        excluded_ids.update([img['id'] for img in image_data if img['path'].startswith('Query/')])

    # Find valid indices excluding the above IDs and 'Query' category
    valid_indices = [i for i, img in enumerate(image_data) if img['id'] not in excluded_ids and not img['path'].startswith('Query/')]

    if len(valid_indices) == 0:
        return jsonify({'error': 'No more images available for retrieval.'}), 400

    valid_features = features_matrix[valid_indices]
    distances = np.linalg.norm(valid_features - updated_query, axis=1)

    # Check if we are beyond the iteration limit -> final retrieval
    if iteration + 1 > iteration_limit:
        # Final retrieval: Consider all images except the query image and 'Query' category, return the user-defined number of images
        all_indices = [i for i, img in enumerate(image_data) if img['id'] != query_image_id and not img['path'].startswith('Query/')]
        if not all_indices:
            return jsonify({'error': 'No images available for final retrieval.'}), 400
        all_features = features_matrix[all_indices]
        all_distances = np.linalg.norm(all_features - updated_query, axis=1)
        top_indices = np.argsort(all_distances)[:final_results_count]
        top_final = [image_data[all_indices[i]] for i in top_indices]
        final_retrieval = [{'path': img['path'], 'id': img['id']} for img in top_final]
        return jsonify({'final_retrieval': final_retrieval})
    else:
        # Intermediate iteration retrieval
        top_indices_within_valid = np.argsort(distances)[:final_results_count]
        top5_indices = [valid_indices[i] for i in top_indices_within_valid]
        top5 = [image_data[i] for i in top5_indices]
        retrieved_images = [{'path': img['path'], 'id': img['id']} for img in top5]

        if 'retrieved_ids' not in session:
            session['retrieved_ids'] = []
        session['retrieved_ids'].extend([img['id'] for img in top5])

        # Increment iteration
        session['current_iteration'] = iteration + 1

        return jsonify({'retrieved_images': retrieved_images, 'iteration': session['current_iteration']})

# ----------------- Feature Visualization Route -----------------

def get_full_image_path(relative_path):
    """
    Determines the full path of an image by checking the RSSCN7-master directory.

    Parameters:
        relative_path (str): The relative path of the image.

    Returns:
        str: The full path to the image if found, else None.
    """
    rsscn7_full_path = os.path.join(app.root_path, PARENT_DIR, relative_path)
    if os.path.exists(rsscn7_full_path):
        return rsscn7_full_path

    # Image not found in RSSCN7-master
    return None

@app.route('/features/<string:image_id>')
@login_required
def show_features(image_id):
    # Find the image entry
    img_entry = next((img for img in image_data if img['id'] == image_id), None)
    if img_entry is None:
        # If image entry is not found, check if it's in the database
        image = Image.query.get(image_id)
        if image is None:
            return "Image not found", 404

        # Extract features directly from the file
        full_image_path = os.path.join(app.root_path, PARENT_DIR, image.path)
        if not os.path.exists(full_image_path):
            return "Image file not found", 404

        try:
            features = extract_features(full_image_path)
        except Exception as e:
            return f"Error extracting features: {str(e)}", 500

        img_entry = {'id': image_id, 'path': image.path, 'features': features}
        image_data.append(img_entry)  # Add to in-memory data

    features = img_entry['features']
    # Define feature slices based on the extraction order
    hist_size = 512
    dominant_size = 9
    gabor_size = 12
    hu_size = 7
    lbp_size = 26
    sift_size = 128

    hist = features[0:hist_size]
    # dominant_colors = features[hist_size:hist_size+dominant_size]  # Not used in visualization
    gabor = features[hist_size+dominant_size:hist_size+dominant_size+gabor_size]
    hu = features[hist_size+dominant_size+gabor_size:hist_size+dominant_size+gabor_size+hu_size]
    lbp = features[hist_size+dominant_size+gabor_size+hu_size:hist_size+dominant_size+gabor_size+hu_size+lbp_size]
    sift = features[hist_size+dominant_size+gabor_size+hu_size+lbp_size:]

    # Generate plots
    plots = {}

    # 1. Histogram (Hue, Saturation, Value)
    hist = hist.reshape((8,8,8))
    H = hist.sum(axis=(1,2))
    S = hist.sum(axis=(0,2))
    V = hist.sum(axis=(0,1))

    # Hue Histogram
    fig, ax = plt.subplots()
    ax.bar(range(8), H, color='r', label='Hue')
    ax.set_title('Hue Histogram')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['Hue Histogram'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Saturation Histogram
    fig, ax = plt.subplots()
    ax.bar(range(8), S, color='g', label='Saturation')
    ax.set_title('Saturation Histogram')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['Saturation Histogram'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Value Histogram
    fig, ax = plt.subplots()
    ax.bar(range(8), V, color='b', label='Value')
    ax.set_title('Value Histogram')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['Value Histogram'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # 2. Dominant Colors (Use the new function instead of normalized features)
    # Convert dominant colors to HEX using the new function
    try:
        # Determine the full path of the image using the helper function
        full_image_path = get_full_image_path(img_entry['path'])
        if not full_image_path:
            raise FileNotFoundError(f"Image {img_entry['path']} not found in RSSCN7-master directory.")
        
        dominant_colors_rgb = get_dominant_colors_original(full_image_path, k=3).flatten()
    except Exception as e:
        print(f"Error extracting dominant colors for visualization: {e}")
        dominant_colors_rgb = np.array([0, 0, 0] * 3)  # Default to black if error occurs

    dominant_colors_hex = []
    print(f"Dominant Colors Raw Values: {dominant_colors_rgb}")  # Debugging statement
    for i in range(0, len(dominant_colors_rgb), 3):
        r, g, b = dominant_colors_rgb[i:i+3]
        # Ensure values are within [0, 255]
        r = int(np.clip(r, 0, 255))
        g = int(np.clip(g, 0, 255))
        b = int(np.clip(b, 0, 255))
        hex_color = rgb_to_hex(r, g, b)
        dominant_colors_hex.append(hex_color)
        print(f"Dominant Color {i//3 + 1}: RGB({r}, {g}, {b}) -> HEX({hex_color})")  # Debugging statement

    # 3. Gabor Features
    frequencies = [0.1, 0.3, 0.5]
    gabor_means = gabor[0::4]
    gabor_vars = gabor[1::4]
    gabor_maxs = gabor[2::4]
    gabor_mins = gabor[3::4]

    fig, ax = plt.subplots()
    ax.plot(frequencies, gabor_means, label='Mean', marker='o')
    ax.plot(frequencies, gabor_vars, label='Variance', marker='o')
    ax.plot(frequencies, gabor_maxs, label='Max', marker='o')
    ax.plot(frequencies, gabor_mins, label='Min', marker='o')
    ax.set_title('Gabor Features')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Value')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['Gabor Features'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # 4. Hu Moments
    fig, ax = plt.subplots()
    ax.bar(range(1,8), hu, color='purple')
    ax.set_title('Hu Moments')
    ax.set_xlabel('Moment')
    ax.set_ylabel('Value')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['Hu Moments'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # 5. LBP Histogram
    fig, ax = plt.subplots()
    ax.bar(range(len(lbp)), lbp, color='grey')
    ax.set_title('LBP Histogram')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['LBP Histogram'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # 6. SIFT Features
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(range(len(sift)), sift, color='orange')
    ax.set_title('SIFT Features')
    ax.set_xlabel('Descriptor Index')
    ax.set_ylabel('Value')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['SIFT Features'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Pass the plots, image path, and dominant colors to the template
    return render_template('features.html', image_path=img_entry['path'], plots=plots, dominant_colors=dominant_colors_hex)

# ----------------- Serve Uploaded Images -----------------

@app.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    # Since all images are now in RSSCN7-master, serve from there
    rsscn7_full_path = os.path.join(app.root_path, PARENT_DIR)
    file_path = os.path.join(rsscn7_full_path, filename)
    if os.path.exists(file_path):
        return send_from_directory(rsscn7_full_path, filename)
    return "File not found", 404

# ----------------- Initialize the Image Database on Startup -----------------

init_image_database()

# ----------------- Run the Flask App -----------------

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True , port= 9000)
