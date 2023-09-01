"""Flask application wrapper for the serving the models."""


import logging
from flask import Flask, request, jsonify
from scoring.score import init, run
app = Flask('app')
init()


@app.route('/healthcheck', methods=['GET'])
def health_check():
    """
    Define an API endpoint that checks the health of the server.

    Returns:
    - (JSON): {"status": "OK"}
    """
    return jsonify({"status": "OK"})


@app.route('/score', methods=['POST'])
def score():
    """
    Define an API endpoint to score input data.

    Parameters:
    - raw_data (bytes): The raw input data to score, sent in the request body.

    Returns:
    - (JSON): The scoring results from the `run()` function.
    """
    return run(raw_data=request.data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
else:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
