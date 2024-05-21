# HandLexa: Harnessing Machine Learning for Real-Time Sign Language Translation

  

In the realm of accessible technology, HandLexa stands out as an innovative solution that leverages machine learning to bridge communication gaps. HandLexa utilizes webcam footage to interpret and translate sign language into comprehensible words, enhancing communication for the deaf and hard-of-hearing community. This blog delves into the technical intricacies of HandLexa, illustrating the step-by-step process it employs to convert images into cohesive words.

  

## Step 1: OpenCV Pipeline

  

The journey begins with image preprocessing using OpenCV, a powerful open-source computer vision library. The primary objective at this stage is to enhance the image in a way that makes it easier for subsequent machine learning models to process and interpret. The OpenCV pipeline for HandLexa involves the following steps:

  

### Grayscale Conversion

  

The initial image is converted from its original color (BGR) to grayscale. This simplifies the image by reducing the complexity of color information, making it easier to highlight essential features such as edges and contours.

  

```python

import cv2 as cv

  

# Load the image

img = cv.imread('hand_sign.jpg')

  

# Convert to grayscale

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

```

  

### Adaptive Thresholding

  

Adaptive thresholding is then applied to the grayscale image. This technique dynamically adjusts the threshold value for each pixel based on its surrounding pixels, creating a binary image where the edges and important features are accentuated.

  

```python

# Apply adaptive thresholding

thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 2)

```

  

### Color Conversion

  

Finally, the binary image is converted back to an RGB format. Although the image remains effectively grayscale, this step ensures compatibility with the TensorFlow model, which expects three-channel input.

  

```python

# Convert back to RGB

processed_img = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)

```

  

The result is a processed image that clearly highlights the hand's shape and contours, ready for the next stage of interpretation.

  

## Step 2: TensorFlow Convolutional Neural Network (CNN)

  

With the preprocessed image in hand, the next step involves using a Convolutional Neural Network (CNN) to classify the hand signs. After experimenting with various model architectures and transfer learning from established networks like VGG19 and AlexNet, the team settled on a modified version of MobileNet for its balance of accuracy and real-time performance.

  

### Model Architecture

  

```python

import tensorflow as tf

from tensorflow.keras import layers, models

  

# Load the pre-trained MobileNet model

base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

  

# Add custom dense layers on top

model = models.Sequential([

base_model,

layers.GlobalAveragePooling2D(),

layers.Dense(1024, activation='relu'),

layers.Dense(512, activation='relu'),

layers.Dense(29, activation='softmax') # Assuming 26 letters + 3 additional signs

])

  

# Compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

```

  

The MobileNet base model extracts rich features from the input images, while the additional dense layers fine-tune these features to classify hand signs accurately. This architecture enables real-time sign language recognition, crucial for a seamless user experience.

  

## Step 3: Interpreting the Results

  

The CNN provides a continuous stream of classified outputs, each corresponding to a detected hand sign. However, translating these individual letters into readable words requires further processing. HandLexa employs a custom algorithm to achieve this:

  

### Letter Stream to Word Conversion Algorithm

  

```python

class HandLexaInterpreter:

def __init__(self):

self.previous_letter = ''

self.letter_count = 0

self.predicted_word = ''

  

def interpret(self, current_letter, confidence, threshold=5):

if current_letter == self.previous_letter:

self.letter_count += 1

else:

if self.letter_count >= threshold:

self.predicted_word += self.previous_letter

self.previous_letter = current_letter

self.letter_count = 1

  

# Return the current word prediction

return self.predicted_word

  

# Usage example

interpreter = HandLexaInterpreter()

stream_of_letters = ['h', 'h', 'h', 'e', 'e', 'l', 'l', 'l', 'l', 'o']

  

for letter in stream_of_letters:

word = interpreter.interpret(letter, confidence=0.9)

print(f"Current predicted word: {word}")

```

  

The repetition threshold (x) is dynamic, varying based on the model's confidence in different letters. For letters the model struggles with, a smaller x value is used, and for more confidently detected letters, a larger x value is applied. This dynamic thresholding ensures a higher accuracy in forming words from the detected letters.

  

## Step 4: Autocorrection and Text-to-Speech

  

Despite the robust pipeline, the output is not always flawless. To address this, HandLexa incorporates autocorrection and text-to-speech features:

  

### Autocorrection

  

The predicted words are passed through an autocorrection algorithm that adjusts them to the nearest valid English words. This step helps mitigate minor errors in the letter detection phase.

  

```python

from spellchecker import SpellChecker

  

spell = SpellChecker()

  

def autocorrect(word):

corrected_word = spell.correction(word)

return corrected_word

  

# Usage example

predicted_word = "helo"

corrected_word = autocorrect(predicted_word)

print(f"Corrected word: {corrected_word}")

```

  

### Text-to-Speech

  

The corrected words are then fed into a browser-based text-to-speech API. This feature provides an audible output of the translated sign language, broadening HandLexa's usability.

  

```python

import pyttsx3

  

def speak(text):

engine = pyttsx3.init()

engine.say(text)

engine.runAndWait()

  

# Usage example

speak(corrected_word)

```

## What our app does:

HandLexa is an innovative app that closes the communication gap faced by members of the Deaf and Hard of Hearing communities, who primarily use American Sign Language (ASL) to communicate. HandLexa uses Artificial Intelligence and Machine Learning to recognize ASL signs and convert them into text and audio, allowing Deaf individuals to use sign language to communicate with those who are unfamiliar with it. HandLexa has the potential to make a positive impact for the approximately 600,000 Deaf individuals in the US.
In this pilot version of HandLexa, Deaf individuals can fingerspell (use ASL alphabet signs to spell out words) words into their phone or laptop camera. The app then uses a machine learning model to predict the letter being signed in each frame of the video. It collects the stream of predicted letters and uses a filtering algorithm to extract the text being signed. It has an autocorrect feature to correct any misspelled words, in case the model misses or wrongly classifies a letter. After the user finishes signing their text, the app reads out the text using text-to-speech.

## What inspired us to create it:

There are approximately 600,000 Deaf people in the US, and more than 1 out of every 500 children is born with hearing loss, according to the National Institute on Deafness and Communication Disorders. These members of the Deaf community primarily use American Sign Language to communicate between themselves, however, with the exception of Deaf individuals and their close friends and family members, most people do not know sign language. This makes it difficult for Deaf individuals to communicate with someone outside of their close circle. Common approaches such as writing out or typing text are slow and interrupt the flow of conversation, making them less than desirable. Through our research, we concluded that there are no released apps available that translate between ASL and English, allowing for natural conversation between a Deaf individual and a hearing individual. As a result, we decided to develop one ourselves, building on our knowledge of machine learning and web app design. We believe that communication is a fundamental human right, and all individuals should be able to effectively and naturally communicate with others in the way they choose, and we sincerely hope that HandLexa helps members of the Deaf community achieve this goal.

## Difficulties we faced programming our app:

The biggest technical difficulty we faced in programming this app was developing and training a machine learning model that could accurately identify hand signs in video frames. We were able to solve this challenge by utilizing an OpenCV edge-detection transformation to give the Convolutional Neural Network a simpler image to extract information from, making it easier for it to “learn” how to classify frames into letters. We also used Transfer Learning (from a pre-trained model known as MobileNetV2) to give the model richer “prior knowledge” and thus a better chance of classifying images correctly. Additionally, we had to sacrifice some accuracy to improve the performance of our model, as our initial revisions were too slow to run in real-time on a cell phone CPU. We compensated for this by introducing the autocorrect feature, allowing the model to miss some letters while still producing the correct text. As a result, we were able to find a good balance between accuracy and speed for our final model, which runs at 15 FPS on the average cell phone and still ensures a satisfactory user experience.

## What we would improve on if we were to make a version 2:

We have many ideas in mind for a second version of our app. Firstly, we would like to upgrade our machine learning model to recognize common ASL words rather than only fingerspelling, which will greatly reduce the time it takes for users to input long words. We would also like to train our model to ignore the background by providing it a more diverse range of samples, allowing it to be effective in many more ad-hoc situations. Additionally, we would like to try to incorporate our app with video conferencing software such as Zoom, which will allow users to communicate naturally on conference calls. Finally, we would like to present our app to a group of Deaf individuals and use their feedback to fine-tune the app.

## Built With

1. Python
2. Tensorflow
3. Keras
4. OpenCV
5. JavaScript
6. HTML/CSS
7. React
8. Next.js
