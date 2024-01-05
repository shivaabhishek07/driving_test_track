from flask import Flask,request,send_file,render_template
import io
from utils import process_video

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    application_number = request.form.get('name')

    # Add your logic to fetch data from the backend based on the application_number
    # Replace the following lines with your actual data retrieval logic
    data = {
        'application_number':application_number,
        'applicant_name': 'John Doe',
        'dob': '2024-01-03',
    }

    return render_template('result_page.html', data=data)

@app.route("/verify_face")
def face_verify():

    face_image = request.files['face']

    # Check if the file is empty
    if face_image.filename == '':
        return jsonify({'error': 'Empty file provided'}), 400

    # Save the video file
    FACE_IMAGE = './data/face_img.jpg'
    face_image.save(FACE_IMAGE)

@app.route("/analyze_video", methods=["POST"])
def analyze_video():

    video_file = request.files['video']

    # Check if the file is empty
    if video_file.filename == '':
        return jsonify({'error': 'Empty file provided'}), 400

    # Save the video file
    SOURCE_VIDEO_PATH = './data/uploaded_video.mp4'
    video_file.save(SOURCE_VIDEO_PATH)

    processed_video = process_video(SOURCE_VIDEO_PATH)

    # Return the processed video as a response
    return send_file(
        io.BytesIO(processed_video),
        mimetype='video/mp4',
        download_name='processed_video.mp4'
    )

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)