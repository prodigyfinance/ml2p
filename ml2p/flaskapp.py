# -*- coding: utf-8 -*-

""" Flask application for serving predictions inside a Amazon SageMaker
    docker container.
"""

import pathlib

from flask import Blueprint, Flask, current_app, jsonify, request


predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/ping')
def ping():
    return ("", 200)


@predict_bp.route('/invocations', methods=["POST"])
def invoke():
    data = request.get_json(force=True)
    return jsonify(current_app.predict(data))


def predict_app(predict, instance_path='.', config=None):
    """Create a prediction server web application."""
    instance_path = pathlib.Path(instance_path).absolute()
    app = Flask(__name__, instance_path=str(instance_path))
    app.register_blueprint(predict_bp)
    app.config.from_object(config)
    app.predict = predict
    return app
