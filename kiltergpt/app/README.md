# KilterGPT REST API service

---
## Quick start

### Setup environment

To run and operate the application, you have to set the environment variables values.

| Variable | Default value |
|---|---|
| BIND_IP | 0.0.0.0 |
| BIND_PORT | 8000 |
| HOSTNAME | default |
| BACKEND_CORS_ORIGINS | * |
| LOGGING_LEVEL | INFO |
| SUPABASE_URL | http://127.0.0.1:54321 |
| SUPABASE_KEY | XxXxXxXx |

Env vars can be defined using `.env` file.

You can find example in `example.env`.

```bash
cp conf/example.env .env
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Run project
```bash
uvicorn kiltergpt.app.main:create_app
```
or you can use `__main__.py` for the local development:
```bash
python3 __main__.py
```

### Run tests
Coming soon ...

## Service structure
```
.
├── README.md
├── __init__.py
├── api
│   ├── __init__.py
│   └── v0    # api version
│       ├── __init__.py
│       └── kilter_gpt.py   # http methods for KilterGPT service
├── config.py
├── db
│   ├── __init__.py
│   ├── client.py   # creates client for supabase operations
│   └── kilter.py   # defines data models used in supabase tables
├── dependencies.py   # dependencies from service&repository layers
├── main.py   # web-app entrypoint
├── models
│   ├── __init__.py
│   └── generation.py   # defines requests/responses schemes for api
├── repository
│   ├── __init__.py
│   └── kilter.py   # operations with storage
└── service
    ├── __init__.py
    └── kilter_gpt.py   # service logic for KilterGPT
```
