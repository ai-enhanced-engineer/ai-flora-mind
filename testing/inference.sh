curl --location 'http://localhost:8888/predict' \
    --header 'Content-Type: application/json' \
    --data '{
        "sepal_length": 1.0,
        "sepal_width": 1.0,
        "petal_length": 1.0,
        "petal_width": 1.0
    }'