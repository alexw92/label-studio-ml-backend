import hmac
import logging
import os
import json

from flask import Flask, request, jsonify, Response

from .response import ModelResponse
from .model import LabelStudioMLBase
from .exceptions import exception_handler

logger = logging.getLogger(__name__)

_server = Flask(__name__)
MODEL_CLASS = LabelStudioMLBase
BASIC_AUTH = None


def init_app(model_class, basic_auth_user=None, basic_auth_pass=None):
    global MODEL_CLASS
    global BASIC_AUTH

    if not issubclass(model_class, LabelStudioMLBase):
        raise ValueError('Inference class should be the subclass of ' + LabelStudioMLBase.__class__.__name__)

    MODEL_CLASS = model_class
    basic_auth_user = basic_auth_user or os.environ.get('BASIC_AUTH_USER')
    basic_auth_pass = basic_auth_pass or os.environ.get('BASIC_AUTH_PASS')
    if basic_auth_user and basic_auth_pass:
        BASIC_AUTH = (basic_auth_user, basic_auth_pass)

    return _server


@_server.route('/custom_inference', methods=['GET', 'POST'])
def custom_inference():
    
    CUSTOM_INFERENCE_SECRET = os.getenv("CUSTOM_INFERENCE_SECRET")
    if request.args.get("secret") != CUSTOM_INFERENCE_SECRET:
        abort(403)    
    FAKE_LABEL_CONF = """<View>  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>  <View><Filter toName="label" minlength="0" name="filter"/><RectangleLabels name="label" toName="image">          <Label value="Carrot" background="#FFA39E"/><Label value="Broccoli" background="#D4380D"/><Label value="Tomato" background="#FFC069"/><Label value="Cucumber" background="#AD8B00"/><Label value="Bell-Pepper" background="#D3F261"/><Label value="Ginger" background="#5CDBD3"/><Label value="Garlic" background="#096DD9"/><Label value="Onion" background="#ADC6FF"/><Label value="Eggplant" background="#9254DE"/><Label value="Pumpkin" background="#F759AB"/><Label value="Cabagge" background="#FFA39E"/><Label value="Salad" background="#D4380D"/><Label value="Spinach" background="#FFC069"/><Label value="Leek" background="#AD8B00"/><Label value="Mushroom" background="#389E0D"/><Label value="Zucchini" background="#5CDBD3"/><Label value="Chilli" background="#096DD9"/><Label value="Cauliflower" background="#ADC6FF"/><Label value="Apple" background="#9254DE"/><Label value="Banana" background="#F759AB"/><Label value="Strawberry" background="#FFA39E"/><Label value="Lime" background="#D4380D"/><Label value="Lemon" background="#FFC069"/><Label value="Avocado" background="#AD8B00"/><Label value="Mango" background="#D3F261"/><Label value="Orange" background="#389E0D"/><Label value="Egg" background="#5CDBD3"/><Label value="Rice" background="#096DD9"/><Label value="Pasta" background="#ADC6FF"/><Label value="Lentils" background="#9254DE"/><Label value="Chickpeas" background="#F759AB"/><Label value="Corn" background="#FFA39E"/><Label value="Beans" background="#D4380D"/><Label value="Peas" background="#FFC069"/><Label value="Tofu" background="#AD8B00"/><Label value="Flour" background="#D3F261"/><Label value="Plantmilk" background="#389E0D"/><Label value="Nuts" background="#5CDBD3"/><Label value="Oil" background="#096DD9"/><Label value="Soysauce" background="#ADC6FF"/><Label value="Canned-Tomato" background="#9254DE"/><Label value="Vinegar" background="#F759AB"/><Label value="Balsamico" background="#FFA39E"/><Label value="Cheese" background="#D4380D"/><Label value="Yoghurt" background="#FFC069"/><Label value="Milk" background="#AD8B00"/><Label value="Butter" background="#D3F261"/><Label value="Curd" background="#389E0D"/><Label value="Skyr" background="#5CDBD3"/><Label value="Potato" background="#FFA39E"/><Label value="Scallion" background="#FFA39E"/></RectangleLabels></View></View>"""
    FAKE_PAYLOAD = { "tasks": [{"id": 6628,"data": {"image": "/data/upload/1/85b5fd06-AgACAgIAAxkBAAIJZmaAM93CSZ0gpgJvNY9Q5ZFhV3MzAAIu4jEbxyYAAUgvs8DvEydxZg_k8ASsrC.jpg"},"meta": {},"created_at": "2024-10-12T11:26:51.794813Z","updated_at": "2024-10-12T11:26:51.794823Z","is_labeled": False,"overlap": 1,"inner_id": 3205,"total_annotations": 0,"cancelled_annotations": 0,"total_predictions": 0,"comment_count": 0,"unresolved_comment_count": 0,"last_comment_updated_at": None,"project": 1,"updated_by": None,"file_upload": 6628,"comment_authors": [],"annotations": [],"predictions": []}],"model_version": "0.0.2","project": "1.1694809921","label_config": FAKE_LABEL_CONF,"params": {"login": None,"password": None,"context": None}}
    data = FAKE_PAYLOAD
    tasks = data.get('tasks')
    label_config = data.get('label_config')
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    params = data.get('params', {})
    context = params.pop('context', {})

    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)
    test_result = model.predict_standalone("test")
    
    return jsonify({'message': 'Hello from the other side '+test_result})


@_server.route('/predict', methods=['POST'])
@exception_handler
def _predict():
    """
    Predict tasks

    Example request:
    request = {
            'tasks': tasks,
            'model_version': model_version,
            'project': '{project.id}.{int(project.created_at.timestamp())}',
            'label_config': project.label_config,
            'params': {
                'login': project.task_data_login,
                'password': project.task_data_password,
                'context': context,
            },
        }

    @return:
    Predictions in LS format
    """
    data = request.json
    # check whats inside to mock it
    print(json.dumps(request.json, indent=2))

    tasks = data.get('tasks')
    label_config = data.get('label_config')
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    params = data.get('params', {})
    context = params.pop('context', {})

    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    # model.use_label_config(label_config)

    response = model.predict(tasks, context=context, **params)

    # if there is no model version we will take the default
    if isinstance(response, ModelResponse):
        if not response.has_model_version():
            mv = model.model_version
            if mv:
                response.set_version(str(mv))
        else:
            response.update_predictions_version()

        response = response.model_dump()

    res = response
    if res is None:
        res = []

    if isinstance(res, dict):
        res = response.get("predictions", response)

    return jsonify({'results': res})


@_server.route('/setup', methods=['POST'])
@exception_handler
def _setup():
    data = request.json
    project_id = data.get('project').split('.', 1)[0]
    label_config = data.get('schema')
    extra_params = data.get('extra_params')
    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    if extra_params:
        model.set_extra_params(extra_params)

    model_version = model.get('model_version')
    return jsonify({'model_version': model_version})


TRAIN_EVENTS = (
    'ANNOTATION_CREATED',
    'ANNOTATION_UPDATED',
    'ANNOTATION_DELETED',
    'START_TRAINING'
)


@_server.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    event = data.pop('action')
    if event not in TRAIN_EVENTS:
        return jsonify({'status': 'Unknown event'}), 200
    project_id = str(data['project']['id'])
    label_config = data['project']['label_config']
    model = MODEL_CLASS(project_id, label_config=label_config)
    result = model.fit(event, data)

    try:
        response = jsonify({'result': result, 'status': 'ok'})
    except Exception as e:
        response = jsonify({'error': str(e), 'status': 'error'})

    return response, 201


@_server.route('/health', methods=['GET'])
@_server.route('/', methods=['GET'])
@exception_handler
def health():
    return jsonify({
        'status': 'UP',
        'model_class': MODEL_CLASS.__name__
    })


@_server.route('/metrics', methods=['GET'])
@exception_handler
def metrics():
    return jsonify({})


@_server.errorhandler(FileNotFoundError)
def file_not_found_error_handler(error):
    logger.warning('Got error: ' + str(error))
    return str(error), 404


@_server.errorhandler(AssertionError)
def assertion_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


@_server.errorhandler(IndexError)
def index_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


def safe_str_cmp(a, b):
    return hmac.compare_digest(a, b)


@_server.before_request
def check_auth():
    if request.path == '/hello':
        return  # Skip auth for this path
    if BASIC_AUTH is not None:

        auth = request.authorization
        if not auth or not (safe_str_cmp(auth.username, BASIC_AUTH[0]) and safe_str_cmp(auth.password, BASIC_AUTH[1])):
            return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})


@_server.before_request
def log_request_info():
    logger.debug('Request headers: %s', request.headers)
    logger.debug('Request body: %s', request.get_data())


@_server.after_request
def log_response_info(response):
    logger.debug('Response status: %s', response.status)
    logger.debug('Response headers: %s', response.headers)
    logger.debug('Response body: %s', response.get_data())
    return response
