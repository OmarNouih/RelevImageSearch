# RelevImageSearch

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Methodology](#methodology)
  - [Feature Extraction](#feature-extraction)
    - [Color Histogram](#color-histogram)
    - [Dominant Colors (K-Means Clustering)](#dominant-colors-k-means-clustering)
    - [Gabor Texture Features](#gabor-texture-features)
    - [Hu Moments](#hu-moments)
    - [Local Binary Patterns (LBP)](#local-binary-patterns-lbp)
    - [SIFT Features](#sift-features)
  - [Query-Point Movement Method](#query-point-movement-method)
    - [Rocchio's Algorithm](#rocchios-algorithm)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Set Up the Virtual Environment](#2-set-up-the-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Download the Dataset](#4-download-the-dataset)
    - [5. Configure Environment Variables](#5-configure-environment-variables)
    - [6. Initialize the Database](#6-initialize-the-database)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [API Endpoints](#api-endpoints)
    - [Descriptor API](#descriptor-api)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

**RelevImageSearch** is a **Content-Based Image Retrieval (CBIR)** system built with Flask. It enables users to:

- **Register and Authenticate**: Secure user registration and login functionalities.
- **Upload Images**: Users can upload images categorized into different classes.
- **Feature Extraction**: Extracts various features from images for effective retrieval.
- **Search with Relevance Feedback**: Implements an iterative search process where user feedback refines search results using Rocchio's Algorithm.
- **Visualize Features**: Provides visual representations of extracted features for better understanding.

The system is designed to handle a large image dataset efficiently and provide accurate search results based on user queries and feedback.

## Features

- **User Authentication**: Secure registration, login, and logout functionalities using Flask-Login and Flask-Bcrypt.
- **Image Upload and Management**: Users can upload images, categorize them, and manage their collection.
- **Advanced Feature Extraction**: Utilizes multiple techniques to extract comprehensive features from images, including color histograms, dominant colors, Gabor textures, Hu moments, Local Binary Patterns (LBP), and SIFT features.
- **Relevance Feedback Search**: Implements Rocchio's Algorithm to iteratively refine search results based on user feedback.
- **RESTful API**: Provides API endpoints for feature descriptors.
- **Feature Visualization**: Generates visual plots of extracted features for analysis.
- **Session Management**: Maintains user sessions using Flask-Session.

## Methodology

### Feature Extraction

Effective image retrieval relies on extracting meaningful features that represent the content of images. This system employs a combination of color, texture, shape, and keypoint features to achieve robust retrieval performance.

#### Color Histogram

- **Description**: Captures the distribution of colors in an image.
- **Implementation**: Converts images to HSV color space and computes a 3D histogram with 8 bins per channel (Hue, Saturation, Value), resulting in a 512-dimensional feature vector.
- **Mathematical Basis**: The histogram counts the frequency of pixel intensities within specified ranges, providing a statistical representation of color distribution.

#### Dominant Colors (K-Means Clustering)

- **Description**: Identifies the most prominent colors in an image.
- **Implementation**: Applies K-Means clustering (k=3) on the RGB pixel values to find cluster centers representing dominant colors.
- **Mathematical Basis**: K-Means minimizes the within-cluster sum of squares, effectively grouping similar colors and identifying central points in color space.

#### Gabor Texture Features

- **Description**: Captures texture information by analyzing frequency and orientation.
- **Implementation**: Applies Gabor filters with multiple frequencies (0.1, 0.3, 0.5) to the grayscale image and extracts summary statistics (mean, variance, max, min) from the filtered images.
- **Mathematical Basis**: Gabor filters are sensitive to specific frequencies and orientations, enabling the extraction of edge and texture information.

#### Hu Moments

- **Description**: Captures shape features invariant to image transformations.
- **Implementation**: Computes Hu Moments from image moments and applies a logarithmic transformation to normalize the values.
- **Mathematical Basis**: Hu Moments are seven invariant moments derived from image moments, providing robust shape descriptors.

#### Local Binary Patterns (LBP)

- **Description**: Captures local texture information.
- **Implementation**: Computes LBP with parameters P=24 and R=3, followed by histogram normalization.
- **Mathematical Basis**: LBP encodes the local neighborhood of each pixel, capturing texture by comparing pixel intensities.

#### SIFT Features

- **Description**: Detects and describes local keypoints in images.
- **Implementation**: Extracts SIFT descriptors (up to 100 keypoints) and computes the mean of these descriptors to form a 128-dimensional feature vector.
- **Mathematical Basis**: SIFT (Scale-Invariant Feature Transform) identifies keypoints and computes descriptors invariant to scale, rotation, and illumination.

### Query-Point Movement Method

The search functionality leverages **relevance feedback** to iteratively refine search results based on user interactions. This process employs **Rocchio's Algorithm** to adjust the query vector, enhancing retrieval accuracy over multiple iterations.

#### Rocchio's Algorithm

- **Objective**: Modify the initial query vector to better represent the user's intent by incorporating feedback on relevant and irrelevant documents.
- **Formula**:

  \[
  \vec{q}_{\text{new}} = \alpha \vec{q}_{\text{original}} + \frac{\beta}{|D_r|} \sum_{\vec{d} \in D_r} \vec{d} - \frac{\gamma}{|D_i|} \sum_{\vec{d} \in D_i} \vec{d}
  \]

  - \( \vec{q}_{\text{new}} \): Updated query vector.
  - \( \vec{q}_{\text{original}} \): Original query vector.
  - \( D_r \): Set of documents (images) marked as relevant by the user.
  - \( D_i \): Set of documents (images) marked as irrelevant by the user.
  - \( \alpha, \beta, \gamma \): Tuning parameters that control the influence of the original query, relevant documents, and irrelevant documents, respectively.

- **Implementation in This Project**:

  - **Alpha (\( \alpha \))**: Weight of the original query vector (set to 1.0).
  - **Beta (\( \beta \))**: Weight of relevant documents (set to 0.75).
  - **Gamma (\( \gamma \))**: Weight of irrelevant documents (set to 0.15).

  After each iteration, the query vector is updated based on user feedback, moving it closer to relevant images and away from irrelevant ones in the feature space. This adjustment enhances the relevance of subsequent search results.

## Installation

### Prerequisites

Before you begin, ensure you have the following:

1. **Python 3.7 or Higher**: Download and install from [python.org](https://www.python.org/downloads/).
2. **Git**: Version control system. Download and install from [git-scm.com](https://git-scm.com/downloads).
3. **Virtual Environment** (Recommended): To manage project dependencies separately.

### Steps

#### 1. Clone the Repository

Open your terminal or PowerShell and run:

```bash
git clone https://github.com/OmarNouih/RelevImageSearch.git
cd RelevImageSearch
```

#### 2. Set Up the Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies

Ensure you have `pip` updated:

```bash
pip install --upgrade pip
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

*If you don't have a `requirements.txt` file yet, you can create one by running:*

```bash
pip freeze > requirements.txt
```

#### 4. Download the Dataset

The project utilizes the RSSCN7 dataset for image retrieval. Follow these steps to download and set it up:

1. **Download the Dataset**:

   - Visit the [RSSCN7 Dataset Repository](https://github.com/palewithout/RSSCN7).
   - Click on the **"Code"** button and select **"Download ZIP"**.
   - Alternatively, you can clone the repository:

     ```bash
     git clone https://github.com/palewithout/RSSCN7.git
     ```

2. **Organize the Dataset**:

   - Ensure that the downloaded dataset is placed in the `RSSCN7-master` directory within your project root.
   - The expected directory structure should look like this:

     ```
     RelevImageSearch/
     ├── app.py
     ├── models.py
     ├── forms.py
     ├── requirements.txt
     ├── RSSCN7-master/
     │   ├── Category1/
     │   ├── Category2/
     │   └── ...
     ├── features/
     ├── uploads/
     ├── flask_session/
     ├── templates/
     │   ├── index.html
     │   ├── login.html
     │   ├── register.html
     │   ├── upload.html
     │   ├── category.html
     │   ├── search.html
     │   └── features.html
     ├── static/
     │   ├── css/
     │   ├── js/
     │   └── images/
     └── README.md
     ```

   - If you cloned the RSSCN7 repository separately, move or copy its contents into the `RSSCN7-master` directory of your project.

#### 5. Configure Environment Variables

Create a `.env` file in the project root to store sensitive information and configuration settings.

```bash
touch .env
```

Add the following variables to `.env`:

```env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your_secure_secret_key
SQLALCHEMY_DATABASE_URI=sqlite:///site.db
```

*Replace `your_secure_secret_key` with a strong, unique key. You can generate one using Python:*

```python
import secrets
secrets.token_hex(16)
```

#### 6. Initialize the Database

Set up the SQLite database and create necessary tables.

1. **Activate the Virtual Environment** (if not already active):

   ```bash
   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Run the Application Once to Initialize the Database**:

   ```bash
   python app.py
   ```

   *This will create the `site.db` file and initialize the database schema.*

3. **Alternatively, Use Flask-Migrate (If Set Up)**:

   ```bash
   flask db init
   flask db migrate -m "Initial migration."
   flask db upgrade
   ```

---

## Usage

### Running the Application

Start the Flask development server:

```bash
python app.py
```

Access the application by navigating to `http://127.0.0.1:5000/` in your web browser.

### Application Workflow

1. **Register and Login**
   - Create a new account or log in with existing credentials.

2. **Upload Images**
   - Navigate to the upload page to add new images categorized appropriately.

3. **Browse Categories**
   - View images categorized into different classes.

4. **Search Images**
   - Upload a query image to initiate a search.
   - Provide feedback by marking retrieved images as relevant or irrelevant.
   - The system refines search results based on your feedback.

5. **Visualize Features**
   - View visual representations of extracted features for any image.

### API Endpoints

#### Descriptor API

- **Endpoint**: `/api/descriptor`
- **Method**: `POST`
- **Description**: Returns the color histogram of the specified image.
- **Request Body**:

  ```json
  {
    "image_path": "path/to/image.jpg"
  }
  ```

- **Response**:

  ```json
  {
    "histogram": [/* 512-dimensional histogram */]
  }
  ```

---

## Requirements

The project relies on the following Python libraries:

- **Flask**: Web framework.
- **Flask-RESTful**: For building REST APIs.
- **Flask-Login**: User session management.
- **Flask-Bcrypt**: Password hashing.
- **Flask-CORS**: Cross-Origin Resource Sharing.
- **Flask-Session**: Server-side session management.
- **Flask-WTF**: Form handling.
- **SQLAlchemy**: ORM for database interactions.
- **OpenCV (cv2)**: Image processing.
- **NumPy**: Numerical computations.
- **scikit-learn**: Machine learning utilities (K-Means).
- **scikit-image**: Image feature extraction (LBP).
- **Matplotlib**: Plotting and visualization.
- **tqdm**: Progress bars.

All dependencies are listed in the `requirements.txt` file and can be installed using `pip`.

---

## Project Structure

```
RelevImageSearch/
├── app.py
├── models.py
├── forms.py
├── requirements.txt
├── RSSCN7-master/
│   ├── Category1/
│   ├── Category2/
│   └── ...
├── features/
├── uploads/
├── flask_session/
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── upload.html
│   ├── category.html
│   ├── search.html
│   └── features.html
├── static/
│   ├── css/
│   ├── js/
│   └── images/
└── README.md
```

- **app.py**: Main Flask application containing routes and logic.
- **models.py**: Database models using SQLAlchemy.
- **forms.py**: WTForms for user registration and login.
- **RSSCN7-master/**: Directory containing the image dataset organized into categories.
- **features/**: Stores extracted feature vectors.
- **uploads/**: Stores user-uploaded images.
- **flask_session/**: Stores session data.
- **templates/**: HTML templates for rendering pages.
- **static/**: Static files like CSS, JavaScript, and images.
- **README.md**: Project documentation.

