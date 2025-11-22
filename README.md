# MNIST Digit Calculator

A Streamlit app that recognizes handwritten digits using a simple TensorFlow neural network (Input → Flatten → Dense) and performs arithmetic operations on the predicted digits.

## Key Features & Benefits

*   **Handwritten Digit Recognition:** Accurately identifies digits drawn on a canvas using a trained neural network.
*   **Arithmetic Operations:** Performs basic arithmetic operations (addition, subtraction, multiplication, division) on recognized digits.
*   **Interactive Interface:** User-friendly Streamlit interface for drawing digits and viewing results.
*   **Pre-trained Model:** Includes a pre-trained MNIST model for immediate use.
*   **Simple Architecture:** Utilizes a basic neural network architecture for easy understanding and modification.

## Prerequisites & Dependencies

Before running the application, ensure you have the following installed:

*   **Python:** (Version 3.7 or higher recommended)
*   **TensorFlow:**
*   **Keras:**
*   **NumPy:**
*   **Streamlit:**
*   **Pillow (PIL):**
*   **streamlit-drawable-canvas:**

Install the required Python packages using pip:

```bash
pip install tensorflow keras numpy streamlit pillow streamlit-drawable-canvas
```

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/MuhammadUsman-Khan/MNIST-Digit-Calculator.git
    cd MNIST-Digit-Calculator
    ```

2.  **Verify dependencies:** Make sure all dependencies from the Prerequisites section are installed correctly.  If not, run the `pip install` command mentioned above.

3.  **Run the Streamlit app:**

    ```bash
    streamlit run mnist.py
    ```

    This command will start the Streamlit server and open the application in your default web browser.

## Usage Examples

1.  **Draw a digit:** Use the drawing canvas to draw a digit from 0 to 9.
2.  **View the prediction:** The application will display the predicted digit based on the neural network's output.
3.  **Perform calculations:** (Currently this functionality is not fully implemented as evident from the original structure). The envisioned functionality involves drawing multiple digits and performing calculations with them.  For example, draw '2' and then '3', and perform 2 + 3.

## Configuration Options

The following aspects can be configured:

*   **Drawing Canvas Parameters:**  The `streamlit_drawable_canvas` component offers configuration for stroke width, color, and background.  These settings can be modified within the `mnist.py` file.
*   **Model Training:** While the repository includes a pre-trained model (`mnist_model.h5`), you can retrain the model (code for training should be added to `train_mnist_model()` in `mnist.py`).

## Contributing Guidelines

We welcome contributions to improve this project! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and ensure they are well-documented.
4.  Submit a pull request with a clear description of your changes.

## License Information

This project has no license specified. All rights are reserved by the owner.

## Acknowledgments

*   This project utilizes the MNIST dataset, which is a widely used dataset for handwritten digit recognition.
*   The Streamlit library provides the foundation for building the interactive web application.
*   The TensorFlow and Keras libraries enable the creation and training of the neural network model.
