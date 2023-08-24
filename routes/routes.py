
from flask import Blueprint, jsonify, request
from services import DetectionService
# import asyncio

bp = Blueprint('routes', __name__, url_prefix='/v1/')

myDetector = DetectionService()

################################################################
@bp.route('/detect-all', methods=['POST'])
def detect_all(): 
    # get the image data from the request
    image_data = request.json['image']
    myDetector.detect(image_data)
    return get_response(myDetector)

################################################################
@bp.route('/detect-face', methods=['POST'])
def detect_face(): 
    # get the image data from the request
    image_data = request.json['image']
    myDetector.detect(image_data, feature="face")
    if not myDetector.is_face:
        return jsonify({'success': False, 'status': 400, 'message': "Not a face."}),400
    elif myDetector.is_multiple_faces:
        return jsonify({'success': False, 'status': 400, 'message': "Multiple faces detected."}),400
    else:
        return jsonify({'success': True, 'status': 200, 'message': 'All good!'}),200
    
################################################################
@bp.route('/detect-hand', methods=['POST'])
def detect_hand(): 
    # get the image data from the request
    image_data = request.json['image']
    myDetector.detect(image_data, feature="hand")
    if myDetector.has_hand:
        return jsonify({'success': False, 'status': 400, 'message': "Hand(s) detected."}),400
    else:
        return jsonify({'success': True, 'status': 200, 'message': 'All good!'}),200

################################################################
@bp.route('/detect-glasses', methods=['POST'])
def detect_glasses(): 
    # get the image data from the request
    image_data = request.json['image']
    myDetector.detect(image_data, feature="glasses")
    if myDetector.has_glasses:
        return jsonify({'success': False, 'status': 400, 'message': "Eye glasses detected."}),400
    else:
        return jsonify({'success': True, 'status': 200, 'message': 'All good!'}),200

################################################################
@bp.route('/detect-expression', methods=['POST'])
def detect_expression(): 
    # get the image data from the request
    image_data = request.json['image']
    myDetector.detect(image_data, feature="expression")
    if not myDetector.is_neutral:
        return jsonify({'success': False, 'status': 400, 'message': "Keep a straight face."}),400
    else:
        return jsonify({'success': True, 'status': 200, 'message': 'All good!'}),200
    
    
################################################################
@bp.route('/detect-face-mask', methods=['POST'])
def detect_face_mask(): 
    # get the image data from the request
    image_data = request.json['image']
    myDetector.detect(image_data, feature="face-mask")
    if not myDetector.has_face_mask:
        return jsonify({'success': False, 'status': 400, 'message': "Face mask detected."}),400
    else:
        return jsonify({'success': True, 'status': 200, 'message': 'All good!'}),200
    
################################################################
@bp.route('/remove-background', methods=['POST'])
def remove_background(): 
    # get the image data from the request
    image_data = request.json['image']
    color = request.json['color']
    format = request.json['format']
    cleared_image = myDetector.clear_background(image_data, color, format)

    return jsonify({'success': True, 'data': cleared_image, 'status': 200, 'message': 'Background cleared.'}),200
    
################################################################  
def get_response(myDetector):
    if not myDetector.is_face:
        return jsonify({'success': False, 'status': 400, 'message': "Not a face."}),400
    elif myDetector.is_multiple_faces:
        return jsonify({'success': False, 'status': 400, 'message': "Multiple faces detected."}),400
    elif myDetector.has_hand:
        return jsonify({'success': False, 'status': 400, 'message': "Keep hands away from face."}),400
    elif myDetector.has_glasses:
        return jsonify({'success': False, 'status': 400, 'message': "Eye glasses detected."}),400
    elif myDetector.has_face_mask:
        return jsonify({'success': False, 'status': 400, 'message': "Face mask detected."}),400
    if not myDetector.is_neutral:
        return jsonify({'success': False, 'status': 400, 'message': "Keep a straight face."}),400
    else:
        return jsonify({'success': True, 'status': 200, 'message': 'All good!'}),200