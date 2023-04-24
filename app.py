import json
import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from routes import routes

cors = CORS()

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )
    register_cors(app)
    register_error_handler(app)
    register_routes(app)
    

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # healthcheck
    @app.route('/')
    def health():
        return jsonify({"message": "Ok!"})
    
    
    return app

def register_cors(app):
    app.logger.info("Registering cors handlers....")
    cors.init_app(app, origins="*")
    
def register_error_handler(app):
    # global handler for uncaught exceptions
    def handle_exception(e):
        logging.exception(e)
        response = jsonify({
            "status": 500,
            "error": "Server error",
            "message": "Something went wrong. Try again.",
        })
        response.content_type = "application/json"
        return response, 500
    
    # global handler for http exceptions
    def handle_error(e):
        response = e.get_response()
        response.data = json.dumps({
            "status": e.code,
            "error": e.name,
            "message": e.description,
        })
        response.content_type = "application/json"
        return response, e.code
    app.errorhandler(HTTPException)(handle_error)
    app.errorhandler(Exception)(handle_exception)

def register_routes(app):
    app.register_blueprint(routes.bp)

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=9000)