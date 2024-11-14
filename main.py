# Core Libraries
import openai  # for NLP responses
import speech_recognition as sr  # for Speech-to-Text (Whisper can replace this)
import pyttsx3  # for Text-to-Speech (or use MelO-TTS API)
from gtts import gTTS  # Fallback Text-to-Speech using Google TTS
import os  # For saving and playing MP3 files
import cv2  # for Face Recognition using OpenCV

# Initial Setup
openai.api_key = "Your Api key"  # Replace with your OpenAI API key
recognizer = sr.Recognizer()
engine = pyttsx3.init(driverName='sapi5')  # For Windows. Use 'espeak' for Linux/macOS.
engine.setProperty('rate', 150)  # Set the speech speed.
engine.setProperty('volume', 1)  # Set the volume level (0.0 to 1.0).

# Load Face Recognition Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function for Thinking/Conversations (NLP)
def get_ai_response(prompt):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",  # or use any other GPT-based model you prefer
            prompt=prompt,
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error in generating AI response: {e}")
        return "Sorry, I couldn't process that request."

# Function for Emotion Recognition (basic text-based detection for now)
def detect_emotion(text):
    if "happy" in text.lower():
        return "happy"
    elif "sad" in text.lower():
        return "sad"
    else:
        return "neutral"

# Function for Speech-to-Text (Listening to the user)
def listen_to_user():
    with sr.Microphone() as source:
        print("Listening for your voice...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)  # You can switch this to Whisper
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return "Sorry, I didn't catch that."

# Function for Text-to-Speech (Responding to the user)
def speak(text):
    print(f"Speaking: {text}")  # Debugging print

    try:
        # Try pyttsx3 first
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error with pyttsx3: {e}")
        print("Falling back to Google TTS (gTTS)...")
        try:
            # If pyttsx3 fails, fall back to gTTS
            tts = gTTS(text=text, lang='en')
            tts.save("response.mp3")
            os.system("start response.mp3")  # For Windows. Use 'open' for macOS and 'mpg321' for Linux.
        except Exception as e:
            print(f"Error with gTTS: {e}")

# Function for Face Recognition
def recognize_face():
    cap = cv2.VideoCapture(0)
    recognized = False
    while not recognized:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if faces != ():
            print("Face detected!")
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            recognized = True
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return "User recognized." if recognized else "Face not recognized."

# Main Loop
def main():
    print("Starting AI...")
    speak("Hello, I am your AI assistant.")
    
    face_status = recognize_face()  # This will run face recognition and return status
    if face_status == "User recognized.":
        print("Welcome! I'm ready to talk.")
        while True:
            user_input = listen_to_user()  # Listen for user input
            if user_input:
                print(f"User said: {user_input}")
                
                # Get AI response to user's input
                response = get_ai_response(user_input)
                print(f"AI Response: {response}")
                
                # Detect Emotion (optional)
                emotion = detect_emotion(response)
                print(f"Emotion detected: {emotion}")
                
                # Respond using text-to-speech
                speak(response)
    else:
        print("Face not recognized. Please try again.")

# Run the AI
if __name__ == "_main_":
    main()