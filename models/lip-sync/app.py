import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse

# A list of characters (alphabet and a few punctuation marks)
# We assume index 0 is reserved for the CTC "blank" token.
characters = "abcdefghijklmnopqrstuvwxyz '"
num_chars = len(characters)

from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers import Input
from keras.models import Model
from lipnet.core.layers import CTC
from keras import backend as K


class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, absolute_max_string_len=32, output_size=28):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        self.build()

    def build(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_w, self.img_h)
        else:
            input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)

        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        self.zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.input_data)
        self.conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), activation='relu', kernel_initializer='he_normal', name='conv1')(self.zero1)
        self.maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.conv1)
        self.drop1 = Dropout(0.5)(self.maxp1)

        self.zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(self.drop1)
        self.conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv2')(self.zero2)
        self.maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(self.conv2)
        self.drop2 = Dropout(0.5)(self.maxp2)

        self.zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(self.drop2)
        self.conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv3')(self.zero3)
        self.maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(self.conv3)
        self.drop3 = Dropout(0.5)(self.maxp3)

        self.resh1 = TimeDistributed(Flatten())(self.drop3)

        self.gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(self.resh1)
        self.gru_2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(self.gru_1)

        # transforms RNN output to character activations:
        self.dense1 = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.gru_2)

        self.y_pred = Activation('softmax', name='softmax')(self.dense1)

        self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=self.loss_out)

    def summary(self):
        Model(inputs=self.input_data, outputs=self.y_pred).summary()

    def predict(self, input_batch):
        return self.test_function([input_batch, 0])[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        return K.function([self.input_data, K.learning_phase()], [self.y_pred, K.learning_phase()])

#############################################
# Video Preprocessing: Extract Mouth Frames
#############################################
def extract_mouth_frames(video_path, num_frames=75):
    """
    Opens the video file and extracts frames.
    For each frame, a face detector (Haar Cascade) locates the face,
    and a heuristic crops out the mouth region.
    The function returns a list of mouth-region images (as NumPy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) > 0:
            # Take the first detected face
            (x, y, w, h) = faces[0]
            # Define a mouth region (this heuristic may be tuned)
            mouth_region = frame[y + int(0.65 * h): y + h, x + int(0.25 * w): x + int(0.75 * w)]
            if mouth_region.size == 0:
                continue
            # Resize the mouth region to a fixed size (e.g., 100x50 pixels)
            mouth_region = cv2.resize(mouth_region, (100, 50))
            frames.append(mouth_region)
    cap.release()
    
    # Ensure we have a fixed number of frames (pad with the last frame or truncate)
    if len(frames) < num_frames:
        if len(frames) == 0:
            raise ValueError("No frames were extracted from the video.")
        last_frame = frames[-1]
        while len(frames) < num_frames:
            frames.append(last_frame)
    else:
        frames = frames[:num_frames]
    return frames

def preprocess_frames(frames):
    """
    Converts a list of mouth-region frames into a PyTorch tensor.
    Output tensor shape: (1, channels, frames, height, width)
    """
    # Stack frames into a NumPy array: (num_frames, height, width, channels)
    frames_array = np.stack(frames, axis=0)
    frames_array = frames_array.astype(np.float32) / 255.0  # normalize pixel values
    # Rearrange dimensions to: (batch, channels, frames, height, width)
    frames_tensor = torch.tensor(frames_array).permute(3, 0, 1, 2).unsqueeze(0)
    return frames_tensor

#############################################
# Greedy CTC Decoder
#############################################
def greedy_ctc_decoder(output, characters):
    """
    Decodes the output of the network using a simple greedy approach.
    - output: Tensor of shape (1, frames, num_classes)
    - characters: The string of valid characters (index 0 is reserved for the blank token).
    
    This decoder:
      1. Picks the highest-probability character at each time step.
      2. Collapses repeated characters.
      3. Removes blank tokens.
    """
    output = output.squeeze(0)  # (frames, num_classes)
    # Pick the most likely character at each time step
    pred_indices = torch.argmax(output, dim=1).cpu().numpy()
    
    transcript = ""
    prev = -1
    for idx in pred_indices:
        # Skip if the index is the same as previous (collapse repeats)
        # and skip if index is 0 (the blank token)
        if idx != prev and idx != 0:
            # Map index to character: we subtract 1 because our characters start at index 1.
            transcript += characters[idx - 1]
        prev = idx
    return transcript

#############################################
# Main Transcription Function
#############################################
def main(video_path):
    # Define number of classes for the model: actual characters + 1 blank token.
    num_classes = num_chars + 1
    # Instantiate the model (in practice, load your pre-trained weights)
    model = LipNet(num_classes)
    
    # Example: load pre-trained weights if available
    model.load_state_dict(torch.load('overlapped-weights368.h5', map_location='cpu'))
    
    model.eval()  # set the model to evaluation mode
    
    # Extract mouth-region frames from the video
    frames = extract_mouth_frames(video_path)
    input_tensor = preprocess_frames(frames)
    
    with torch.no_grad():
        # Forward pass: output shape (1, frames, num_classes)
        output = model(input_tensor)
    
    # Decode the network output to obtain the transcript
    transcript = greedy_ctc_decoder(output, characters)
    print("Transcription:", transcript)

