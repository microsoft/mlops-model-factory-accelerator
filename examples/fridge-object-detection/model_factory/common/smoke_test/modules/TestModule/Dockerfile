FROM python:3.8-slim-buster

WORKDIR /app

COPY model_factory/common/smoke_test/modules/TestModule/requirements.txt ./
RUN pip install -r requirements.txt

COPY ./model_factory/common/smoke_test/ ./model_factory/common/smoke_test
COPY ./model_factory/nyc_taxi/smoke_test/ ./model_factory/nyc_taxi/smoke_test
COPY ./model_factory/vehicle_obj_det/smoke_test/ ./model_factory/vehicle_obj_det/smoke_test
COPY ./model_factory/fridge_obj_det/smoke_test/ ./model_factory/fridge_obj_det/smoke_test

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD [ "python3", "-u", "./model_factory/common/smoke_test/modules/TestModule/main.py" ]