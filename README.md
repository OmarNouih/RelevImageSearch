# RelevImageSearch

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Methodology](#methodology)
  - [Feature Extraction](#feature-extraction)
    - [Color Histogram](#color-histogram)
    - [Dominant Colors (K-Means Clustering)](#dominant-colors-k-means-clustering)
    - [Gabor Texture Features](#gabor-texture-features)
    - [Hu Moments (Shape Features)](#hu-moments-shape-features)
    - [Local Binary Patterns (LBP)](#local-binary-patterns-lbp)
    - [SIFT Features](#sift-features)
  - [Relevance Feedback (Query-Point Movement)](#relevance-feedback-query-point-movement)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Set Up the Virtual Environment](#2-set-up-the-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Download the Dataset](#4-download-the-dataset)
    - [5. Initialize the Database](#5-initialize-the-database)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Application Workflow](#application-workflow)
  - [API Endpoints](#api-endpoints)
    - [Descriptor API](#descriptor-api)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Contributing](#contributing)


## Overview

**RelevImageSearch** is a **Content-Based Image Retrieval (CBIR)** system built with Flask. It allows users to securely register, upload images, and perform image searches based on visual content rather than text keywords. The system incorporates a range of feature extraction techniques and a feedback-driven search refinement process to deliver more accurate and relevant search results over time.

## Features

- **User Authentication**: Secure registration, login, and logout functionalities.
- **Image Upload and Management**: Users can upload and categorize images.
- **Advanced Feature Extraction**: Extracts diverse characteristics from images, including color, texture, shape, and local keypoints.
- **Relevance Feedback**: Iteratively refines search results based on user feedback.
- **RESTful API**: Provides endpoints for retrieving image descriptors.
- **Feature Visualization**: Offers insights into how images are represented internally.
- **Session Management**: Uses Flask-Session for maintaining user sessions.

## Methodology

### Feature Extraction

Effective image retrieval depends on extracting meaningful features that represent an image’s visual content. By combining different feature types, we can capture various aspects of an image and improve the reliability of retrieval.

#### Color Histogram

Summarizing the color distribution helps represent the overall color composition of an image. By using histograms that count how often certain color ranges occur, we can quickly compare images based on their predominant hues and intensities.  

#### Dominant Colors (K-Means Clustering)

Instead of considering all colors, we identify a small set of the most common colors. K-Means clustering groups similar colors and picks representative ones, simplifying comparisons.  

#### Gabor Texture Features

Gabor filters help capture the texture patterns in an image. By responding to edges, frequencies, and orientations, these filters provide a way to quantify the image’s “feel” or surface patterns.  

#### Hu Moments (Shape Features)

Hu Moments provide a set of shape descriptors that remain consistent despite rotations, scaling, or reflections. This allows us to characterize the structural layout of an image’s content.  

#### Local Binary Patterns (LBP)

LBP captures small-scale texture by comparing the intensity of each pixel to its neighbors. This simple, yet powerful approach is robust to illumination changes and helps characterize local texture details.  

#### SIFT Features

Scale-Invariant Feature Transform (SIFT) finds keypoints in an image that remain recognizable even when the image is resized, rotated, or viewed from different angles. These keypoints help match images based on distinctive local details.  

### Relevance Feedback (Query-Point Movement)

After the system returns initial search results, users can mark images as relevant or irrelevant. This feedback is then used to adjust the internal representation of the user’s query, guiding the system toward images that better match the user’s criteria. Over multiple rounds, this interactive process fine-tunes the results, improving the precision of the search.  
**Reference:** [Salton, G. & Buckley, C. "Improving retrieval performance by relevance feedback." JASIS (1990)](https://doi.org/10.1002/(SICI)1097-4571(199006)41:4<288::AID-ASI8>3.0.CO;2-H)

## Installation

### Prerequisites

- **Python 3.11 or Higher**: [python.org/downloads](https://www.python.org/downloads/)  
- **Git**: [git-scm.com/downloads](https://git-scm.com/downloads)  
- **Virtual Environment** (Recommended)

### Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/OmarNouih/RelevImageSearch.git
cd RelevImageSearch
```

#### 2. Set Up the Virtual Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Download the Dataset

The project uses the RSSCN7 dataset:

- Visit the [RSSCN7 Dataset Repository](https://github.com/palewithout/RSSCN7)
- Download or clone it, and place the contents in `RSSCN7-master` within the project root.


#### 5. Initialize the Database

```bash
python app.py
```

This creates `site.db` and initializes the schema.

## Usage

### Running the Application

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

### Application Workflow

1. **Register/Login**: Create an account or log in with existing credentials.
2. **Upload Images**: Add images organized by categories.
3. **Browse and Search**: Initiate a search using a query image.
4. **Relevance Feedback**: Mark results as relevant or irrelevant to refine future searches.
5. **Feature Visualization**: Explore extracted features for deeper insight.

### API Endpoints

#### Descriptor API

- **Endpoint**: `/api/descriptor`
- **Method**: `POST`
- **Request**:
  ```json
  {
    "image_path": "path/to/image.jpg"
  }
  ```
- **Response**:
  ```json
  {
    "histogram": [/* ... */]
  }
  ```

## Requirements

- **Flask**, **Flask-RESTful**, **Flask-Login**, **Flask-Bcrypt**, **Flask-CORS**, **Flask-Session**, **Flask-WTF**  
- **SQLAlchemy**  
- **OpenCV (cv2)**  
- **NumPy**  
- **scikit-learn**, **scikit-image**  
- **Matplotlib**  
- **tqdm**

Install all requirements using:

```bash
pip install -r requirements.txt
```

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

## Contributing

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit and push your changes.
4. Open a pull request.

*All GitHub docs are open source. See something that's wrong or unclear? Submit a pull request.*

**© 2024 GitHub, Inc.**  
**[Terms](https://github.com/github/site-policy/blob/main/TERMS.md)** | **[Privacy](https://github.com/github/site-policy/blob/main/PRIVACY.md)** | **[Status](https://www.githubstatus.com/)** | **[Pricing](https://github.com/pricing)** | **[Expert services](https://services.github.com/)** | **[Blog](https://github.blog/)**
