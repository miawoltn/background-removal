
from flask import Blueprint, jsonify, request
from services import DetectionService

bp = Blueprint('routes', __name__, url_prefix='/')

myDetector = DetectionService()

@bp.route('/', methods=['POST'])
def detect_hands():
    # get the image data from the request
    image_data = request.json['image']
    
    myDetector.detect(image_data)
    myDetector.__detect_face()
    
    # return the results as a JSON object
    return get_response(myDetector)

@bp.route('/detect-face', methods=['POST'])
def detect_face(): 
    # get the image data from the request
    image_data = request.json['image']
    
    myDetector.detect(image_data)
    
    return get_response(myDetector)
    
    
def get_response(myDetector):
    if not myDetector.is_face:
        return jsonify({'success': False, 'status': 400, 'message': "Not a face."}),400
    elif myDetector.is_multiple_faces:
        return jsonify({'success': False, 'status': 400, 'message': "Multiple faces detected."}),400
    elif myDetector.has_hand:
        return jsonify({'success': False, 'status': 400, 'message': "Keep hands away from face."}),400
    else:
        return jsonify({'success': True, 'status': 200, 'message': 'All good!'}),200